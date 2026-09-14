"""sgd_bridge_mnist_0914 launcher (copied from snake_phase_mnist_0914 incl. addendum 1 swap rule; stage-1 G1 replaced by the pilot gate) with memory watch (spec §8, §11 steps 4-6).  Standard library only.

    python3 analysis/snake_phase_mnist_0914/launch.py            # main run (refuses unless checks pass & pushed)
    python3 analysis/snake_phase_mnist_0914/launch.py --selftest # S15b

P = min(C_free, floor((MemAvailable - dM_desk) / RSS_peak)); swap not counted.
Watch every 2 s:  MemAvailable < RSS_peak + dM_desk -> SIGSTOP newest running shard, stop launching;
                  MemAvailable < RSS_peak           -> SIGTERM newest shard, move output to _killed/, requeue;
                  MemAvailable >= 2 RSS_peak + dM_desk -> SIGCONT stopped shards (last stopped first).
SIGSTOP frees no memory; it only prevents the stopped shard from growing further.
"""
import argparse, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/sgd_bridge_mnist_0914'
RUNS = OUT / 'runs'
RES = ROOT / 'results/_smoke_sgd_bridge_mnist_0914/resources/resources.json'
PILOT = ROOT / 'results/_smoke_sgd_bridge_mnist_0914/pilot/pilot.json'
MEMLOG = Path('/tmp/claude-1000/-home-issan-Projects-claude/2ffc1715-3da4-4403-919f-af27962a071a/scratchpad/memlog.txt')
ARMS = ['N06', 'P06', 'V06', 'LIN', 'LR']
LR_BOX = 0.02
KB = 1024


def read_meminfo():
    d = {}
    for line in open('/proc/meminfo'):
        k, v = line.split(':', 1)
        d[k] = int(v.split()[0])
    return d['MemAvailable'], d['SwapFree']


def rss_kb(pid):
    try:
        for line in open(f'/proc/{pid}/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def proc_state(pid):
    try:
        return open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[0]
    except OSError:
        return 'X'


def delta_m_desk(memlog, window_s, own_field=4):
    """max over t of (max_{t-w<=tau<=t} M(tau) - M(t)), M = MemAvailable + own shards' RSS (kB)."""
    if not Path(memlog).exists():
        return None
    ts, M = [], []
    for line in open(memlog):
        f = line.split()
        if len(f) < 5:
            continue
        ts.append(int(f[0])); M.append(int(f[1]) + int(f[own_field]))
    best, dq = 0, []                      # monotone deque of indices with decreasing M
    for i, (t, m) in enumerate(zip(ts, M)):
        while dq and ts[dq[0]] < t - window_s:
            dq.pop(0)
        while dq and M[dq[-1]] <= m:
            dq.pop()
        dq.append(i)
        best = max(best, M[dq[0]] - m)
    return best


def busy_other_procs(own_pids):
    try:
        out = subprocess.run(['ps', '-eo', 'pid,pcpu'], capture_output=True, text=True).stdout.split('\n')[1:]
    except Exception:
        return 0
    n = 0
    for line in out:
        f = line.split()
        if len(f) == 2 and float(f[1]) > 50 and int(f[0]) not in own_pids:
            n += 1
    return n


class Launcher:
    def __init__(self, rss_peak_kb, window_s, log_path, meminfo=read_meminfo, spawn=None, memlog=MEMLOG,
                 mut=None, clock=time.time, busy=busy_other_procs):
        self.rss_peak, self.window_s, self.meminfo, self.memlog = rss_peak_kb, window_s, meminfo, memlog
        self.spawn = spawn or self._spawn
        self.mut = mut or {}
        self.running = []              # list of dicts: shard, proc, start, stopped
        self.stop_order = []
        self.log_path = Path(log_path); self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock, self.busy = clock, busy
        self.dM = 0; self.last_dm = -1e9; self.last_start = -1e9; self.last_stop = -1e9
        self.attempts, self.final = {}, {}

    def log(self, msg):
        mem, swap = self.meminfo()
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} mem_avail_kb={mem} swap_free_kb={swap} dM={self.dM} {msg}"
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    def _spawn(self, shard):
        arm, seed = shard
        out = RUNS / f'{arm}_s{seed}'
        logs = OUT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        fh = open(logs / f'{arm}_s{seed}.log', 'w')
        return subprocess.Popen(['nice', '-n', '5', sys.executable, '-m', 'src.sgd_bridge_mnist_0914', '--arm', arm, '--lr', str(LR_BOX),
                                 '--seed', str(seed), '--tasks', '120', '--out', str(out)],
                                cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)

    def refresh_dm(self):
        if self.clock() - self.last_dm < 60:
            return
        v = delta_m_desk(self.memlog, self.window_s)
        if v is not None:
            self.dM = v
        self.last_dm = self.clock()

    def allowed_new(self, mem, swap):
        # addendum 1: stale swap pages do not refill without swapoff, so SwapFree alone blocked all starts
        # for an hour with MemAvailable at 16-20 GiB.  The swap guard applies only when MemAvailable is
        # itself below the resume bound 2 RSS_peak + dM_desk (the same arithmetic as the watch).
        if (not self.mut.get('ignore_swap')) and swap < self.rss_peak and mem < 2 * self.rss_peak + self.dM:
            return 0
        headroom = sum(max(0, self.rss_peak - rss_kb(r['proc'].pid)) for r in self.running)
        by_mem = math.floor((mem - self.dM - headroom) / self.rss_peak)
        c_free = min(20, 28 - self.busy({r['proc'].pid for r in self.running})) - len(self.running)
        return max(0, min(c_free, by_mem))

    def watch_once(self):
        mem, swap = self.meminfo()
        bound1 = self.rss_peak + self.dM
        resume = 2 * self.rss_peak + self.dM
        below1 = (mem > bound1) if self.mut.get('swap_ops') else (mem < bound1)
        below2 = (mem > self.rss_peak) if self.mut.get('swap_ops') else (mem < self.rss_peak)
        alive = [r for r in self.running if r['proc'].poll() is None]
        if below2 and alive:
            r = alive[0] if self.mut.get('oldest') else alive[-1]
            if r['stopped']:
                os.kill(r['proc'].pid, signal.SIGCONT)
            r['proc'].terminate()
            r['killed'] = True
            self.log(f"SIGTERM pid={r['proc'].pid} shard={r['shard']}")
            return 'term'
        if below1:
            cand = [r for r in alive if not r['stopped']]
            if cand and self.clock() - self.last_stop >= 10:
                r = cand[0] if self.mut.get('oldest') else cand[-1]
                os.kill(r['proc'].pid, signal.SIGSTOP); r['stopped'] = True; self.stop_order.append(r)
                self.last_stop = self.clock()
                self.log(f"SIGSTOP pid={r['proc'].pid} shard={r['shard']}")
                return 'stop'
            return 'hold'
        if mem >= resume and self.stop_order:
            r = self.stop_order.pop()
            if r['proc'].poll() is None:
                os.kill(r['proc'].pid, signal.SIGCONT)
            r['stopped'] = False
            self.log(f"SIGCONT pid={r['proc'].pid} shard={r['shard']}")
            return 'cont'
        return 'ok'

    def reap(self, queue):
        for r in list(self.running):
            rc = r['proc'].poll()
            if rc is None:
                continue
            self.running.remove(r)
            if r in self.stop_order:
                self.stop_order.remove(r)
            shard = r['shard']; out = RUNS / f'{shard[0]}_s{shard[1]}'
            status = None
            pj = out / 'provenance.json'
            if pj.exists():
                status = json.loads(pj.read_text()).get('status')
            if rc == 0 and status == 'COMPLETE':
                self.final[shard] = 'COMPLETE'
            elif rc == 3 and status == 'DIVERGED':
                self.final[shard] = 'DIVERGED'
            else:
                ts = time.strftime('%Y%m%d_%H%M%S')
                if out.exists():
                    dst = OUT / '_killed' / ts / out.name
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(out), str(dst))
                self.attempts[shard] = self.attempts.get(shard, 0) + (0 if r.get('killed') else 1)
                if self.attempts[shard] >= 2:
                    self.final[shard] = 'FAILED'
                    self.log(f'FAILED twice shard={shard} rc={rc}')
                else:
                    queue.insert(0, shard)
                    self.log(f'requeue shard={shard} rc={rc} killed={bool(r.get("killed"))}')
            self.log(f'done shard={shard} rc={rc} status={self.final.get(shard)}')

    def run_queue(self, queue, poll=2.0):
        queue = list(queue)
        while queue or self.running:
            self.refresh_dm()
            act = self.watch_once()
            self.reap(queue)
            if act == 'ok' and queue and not self.stop_order and self.clock() - self.last_start >= 5:
                mem, swap = self.meminfo()
                if self.allowed_new(mem, swap) >= 1:
                    shard = queue.pop(0)
                    proc = self.spawn(shard)
                    self.running.append(dict(shard=shard, proc=proc, start=self.clock(), stopped=False))
                    self.last_start = self.clock()
                    self.log(f'start pid={proc.pid} shard={shard} running={len(self.running)} pending={len(queue)}')
            time.sleep(poll)


def validate_table(rows):
    for arm, seed, out in rows:
        if '_smoke' in str(out):
            raise SystemExit(f'refuse: smoke output dir in main table: {out}')
        if arm not in ARMS or not (0 <= seed <= 9):
            raise SystemExit(f'refuse: bad shard {arm} {seed}')


def preflight():
    cj = HERE / 'checks.json'
    if not cj.exists() or not json.loads(cj.read_text()).get('all_pass'):
        raise SystemExit('refuse: checks.json all_pass is not true')
    st = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'src', 'analysis', 'specs', 'configs'],
                        capture_output=True, text=True).stdout.strip()
    if st:
        raise SystemExit(f'refuse: uncommitted changes:\n{st}')
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    up = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '@{u}'], capture_output=True, text=True).stdout.strip()
    if head != up:
        raise SystemExit('refuse: HEAD not pushed')
    if not RES.exists():
        raise SystemExit('refuse: resources.json missing')
    return json.loads(RES.read_text())


def main_run():
    global LR_BOX
    res = preflight()
    if not PILOT.exists():
        raise SystemExit('refuse: pilot.json missing')
    pil = json.loads(PILOT.read_text())
    if pil.get('decision') not in ('GO', 'GO_LR_0.01'):
        raise SystemExit(f"refuse: pilot decision {pil.get('decision')}")
    LR_BOX = 0.01 if pil['decision'] == 'GO_LR_0.01' else 0.02
    table = [(a, s, RUNS / f'{a}_s{s}') for s in range(10) for a in ARMS]
    validate_table(table)
    done = lambda a, s: (RUNS / f'{a}_s{s}/provenance.json').exists() and \
        json.loads((RUNS / f'{a}_s{s}/provenance.json').read_text()).get('status') in ('COMPLETE', 'DIVERGED')
    L = Launcher(res['rss_peak_kb'], res['wall_seconds'], OUT / 'launch_log.txt')
    L.log(f"launcher start rss_peak_kb={res['rss_peak_kb']} window_s={res['wall_seconds']} lr={LR_BOX}")
    L.run_queue([(a, s) for a, s, _ in table if not done(a, s)])
    v = subprocess.run([sys.executable, str(HERE / 'verdict.py'), '--runs', str(RUNS), '--out', str(OUT)], cwd=ROOT,
                       capture_output=True, text=True)
    L.log(f'verdict rc={v.returncode} {v.stdout.strip()[-200:]} {v.stderr.strip()[-500:]}')
    final = {f'{a}_s{s}': st for (a, s), st in L.final.items()}
    (OUT / 'launch_done.json').write_text(json.dumps(dict(final=final, verdict_rc=v.returncode, lr=LR_BOX), indent=1))


# ------------------------------------------------------------------ selftest (S15b)
def selftest():
    import tempfile
    cases, muts = [], []
    tmp = Path(tempfile.mkdtemp())
    G = 1024 * 1024       # 1 GiB in kB

    def mk(mut=None, seq=None, busy=0):
        state = dict(mem=13 * G, swap=4 * G)
        L = Launcher(1 * G, 60, tmp / 'log.txt', meminfo=lambda: (state['mem'], state['swap']),
                     spawn=lambda shard: subprocess.Popen(['sleep', '1000']), memlog=tmp / 'none', mut=mut,
                     clock=lambda: 1e12 + len(L.log_path.read_text()) if L.log_path.exists() else 1e12,
                     busy=lambda own: busy)
        L.dM = G // 2
        return L, state

    def scenario(mut=None):
        L, st = mk(mut)
        L.running = [dict(shard=('A', i), proc=subprocess.Popen(['sleep', '1000']), start=i, stopped=False) for i in range(3)]
        L.last_stop = -1e12
        out = {}
        st['mem'] = int(1.2 * G); out['a1'] = L.watch_once(); time.sleep(0.2)
        out['stopped_pid_is_newest'] = proc_state(L.running[-1]['proc'].pid) == 'T'
        out['oldest_running'] = proc_state(L.running[0]['proc'].pid) in ('S', 'R')
        st['mem'] = int(0.8 * G); out['a2'] = L.watch_once(); time.sleep(0.3)
        out['newest_terminated'] = L.running[-1]['proc'].poll() is not None
        out['others_alive'] = L.running[0]['proc'].poll() is None
        L.running = [r for r in L.running if r['proc'].poll() is None]
        L.stop_order = [r for r in L.stop_order if r['proc'].poll() is None]
        st['mem'] = int(1.2 * G); L.last_stop = -1e12; out['a3'] = L.watch_once(); time.sleep(0.2)
        st['mem'] = int(2.6 * G); out['a4'] = L.watch_once(); time.sleep(0.2)
        out['resumed'] = all(proc_state(r['proc'].pid) != 'T' for r in L.running)
        for r in L.running:
            try:
                os.kill(r['proc'].pid, signal.SIGCONT); r['proc'].kill(); r['proc'].wait()
            except ProcessLookupError:
                pass
        L.running, L.stop_order = [], []
        out['start_blocked_by_swap'] = L.allowed_new(int(2.2 * G), int(0.5 * G)) == 0
        out['swap_ignored_when_mem_ample'] = L.allowed_new(13 * G, int(0.5 * G)) >= 1
        for r in L.running:
            try:
                os.kill(r['proc'].pid, signal.SIGCONT); r['proc'].kill(); r['proc'].wait()
            except ProcessLookupError:
                pass
        return out

    o = scenario()
    cases.append(dict(name='sigstop_newest_below_bound1', ok=o['a1'] == 'stop' and o['stopped_pid_is_newest'] and o['oldest_running']))
    cases.append(dict(name='sigterm_newest_below_rss_peak', ok=o['a2'] == 'term' and o['newest_terminated'] and o['others_alive']))
    cases.append(dict(name='sigcont_at_resume_bound', ok=o['a3'] == 'stop' and o['a4'] == 'cont' and o['resumed']))
    cases.append(dict(name='swap_blocks_start_when_mem_low', ok=o['start_blocked_by_swap']))
    cases.append(dict(name='stale_swap_does_not_block_when_mem_ample', ok=o['swap_ignored_when_mem_ample']))
    L, st = mk()
    L.rss_peak = 20 * G
    cases.append(dict(name='refuse_rss20GiB_P0', ok=L.allowed_new(13 * G, 4 * G) == 0))
    L, st = mk()
    cases.append(dict(name='allows_start_normal', ok=L.allowed_new(13 * G, 4 * G) >= 1))
    try:
        validate_table([('N06', 0, ROOT / 'results/_smoke_sgd_bridge_mnist_0914/x')]); cases.append(dict(name='refuse_smoke_row', ok=False))
    except SystemExit:
        cases.append(dict(name='refuse_smoke_row', ok=True))
    # dM_desk on a synthetic memlog: a 2 GiB dip within the window, own RSS rising compensates 1 GiB
    ml = tmp / 'memlog.txt'
    with open(ml, 'w') as f:
        for i, (ma, own) in enumerate([(10, 0), (10, 0), (8, 0), (9, 1), (10, 0)]):
            f.write(f'{1000 + 2 * i} {ma * G} {G} 0 {own * G}\n')
    cases.append(dict(name='delta_m_desk', ok=delta_m_desk(ml, 100) == 2 * G))
    # mutations
    om = scenario({'swap_ops': True})
    muts.append(dict(name='swap_comparison_ops', detected=not (om['a1'] == 'stop' and om['stopped_pid_is_newest'])))
    oo = scenario({'oldest': True})
    muts.append(dict(name='stop_oldest', detected=not oo['stopped_pid_is_newest']))
    os_ = scenario({'ignore_swap': True})
    muts.append(dict(name='ignore_swap', detected=not os_['start_blocked_by_swap']))
    allp = all(c['ok'] for c in cases) and all(m['detected'] for m in muts)
    (HERE / 'launch_selftest.json').write_text(json.dumps(dict(all_pass=allp, cases=cases, mutations=muts), indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print('launch selftest all_pass', allp, cases, muts)
    return allp


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    main_run()

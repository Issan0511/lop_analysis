"""act_chimera_launch_0913: ジョブキューと進捗ゲート（spec_act_chimera_0913 §6・§7・§8）。

    /usr/bin/python3 -m src.act_chimera_launch_0913 g0                       # G0: 静的検査 → checks/static_checks.json
    /usr/bin/python3 -m src.act_chimera_launch_0913 smoke --env pmnist       # スモーク（箱 B → SCR → RL の順・同時にしない）
    /usr/bin/python3 -m src.act_chimera_launch_0913 smoke --env scr
    /usr/bin/python3 -m src.act_chimera_launch_0913 smoke --env rlmnist --device cpu|cuda
    /usr/bin/python3 -m src.act_chimera_launch_0913 g05                      # G0.5: 判定器の通し（スモーク出力・環境ごとの変異対照）
    /usr/bin/python3 -m src.act_chimera_launch_0913 stage1 --env pmnist [--dry-run]
    /usr/bin/python3 -m src.act_chimera_launch_0913 stage1 --env scr
    /usr/bin/python3 -m src.act_chimera_launch_0913 stage1 --env rlmnist --device cpu|cuda [--parallel J]
    /usr/bin/python3 -m src.act_chimera_launch_0913 scr-lr0p005                # §2.4 の退避枝（起動条件は判定器の LR_FALLBACK_TRIGGER）
    /usr/bin/python3 -m src.act_chimera_launch_0913 stage2 --env pmnist|rlmnist
    /usr/bin/python3 -m src.act_chimera_launch_0913 report [--pass 1|2|both]
    /usr/bin/python3 -m src.act_chimera_launch_0913 backup [--execute]       # CLAUDE.md §4 の退避と backup_manifest.json
    /usr/bin/python3 -m src.act_chimera_launch_0913 status

§7 の規則
- 1 ジョブ = 1 (腕, seed)（SCR は 1 腕）。**各ジョブの起動直前に** ``/proc/meminfo`` の MemAvailable を読み、
  J = ⌊MemAvailable / (1.5 × RSS_peak)⌋ − 1 を計算し直し、実行中の数が J 未満のときだけ起動する（``run_queue``）。
  1.5 は edge_law の ``RSS_HEADROOM``、−1 は見積もりを外したジョブ 1 本の余裕。上限は箱 B 6・SCR 12・RL は処理量で決める。
  MemAvailable は実行中のジョブの分をすでに引いた値なので、2 巡目以降の J は §7 の見積もりより小さく出る（安全側・
  spec の式どおり。§7 の時間の見積もりはこの分だけ遅くなりうる）。
- 固定のプロセス配置はしない（腕ごとに seed を順に回さない）。MemAvailable < 1.5 × RSS_peak なら次のジョブを起動しない。
  実行中のジョブは殺さない。止めるときは、status の ``pgid`` にプロセスグループ単位で送る（``kill -TERM -<pgid>``。
  ジョブは ``start_new_session`` で起こすので /usr/bin/time と学習プロセスが同じグループにいる）。``pkill -f`` は使わない。
- どれかのジョブが非 0 で終わったら、残りのジョブは起動しない（``stopped_after_failure``・実行中のものは殺さない）。
- 時間上限は assert にしない。スモークから外挿した時間の 3 倍を超えたら警告を出すだけ。
- 段は同時に走らせない: サブコマンド全体（後処理の待ち行列を含む）で 1 つの ``results/.act_chimera_0913_active.lock`` を
  O_EXCL で取り、launcher の PID と子の PID がどれか生きていれば別の launcher（同じ段も含む）を起動しない。
- 各ジョブの PID（/usr/bin/time）・子の PID（学習プロセス）・pgid・起動時刻・コマンド行・returncode・wall・ピーク RSS
  （``/usr/bin/time -v``）を ``<launch>/<stage>_status.json`` に逐次書き、段の前後の ``ps`` と ``nvidia-smi`` を
  ``<launch>/<stage>_provenance.json`` に取る。``<launch>`` はスモークなら ``results/_smoke_act_chimera_0913/launch``、
  本走なら ``results/act_chimera_0913/launch``（スモークの記録を本走の結果に混ぜない）。
- RSS_peak は**実測**から取る（箱 B: smoke_summary.json の peak_rss_kib、SCR: 自前の rss_probe の地平線ごとの外挿、
  RL: スモークの provenance と /usr/bin/time のうち ``--device`` の走だけ・追補 2-3）。本走で実測が無ければ ``--rss-peak-gib`` を明示しない限り起動しない。
  スモーク自身は §7 の参考実測（箱 B 0.914 GiB・SCR 0.771 GiB・RL 1.6 GiB）を明示の値として使い、status に出所を書く。
- RL の並列数はスモークの処理量で決める（``rl_throughput``: 並列 1・2・4・J で、各プロセスの provenance の step/s を
  足した処理量を測り、最良の 90% 以内に入る最小の並列数。p 本が実際に同時に走ったことを事象から確かめる）。

進捗ゲート（§6・§7 の停止規則・``require_gates``）
- 段 1 の前: static_checks.json の G0（/usr/bin/python3 と SCR の .venv の両方）と G0.5 が pass で、G0 の
  ``code_sha256``（共有モジュール・ランナー・判定器・launcher・2 テストファイル）と G0.5 の ``report_sha256`` が
  いまのファイルと一致し、G0 の ``git_head`` が HEAD であること（2-4 S6・A7）。
- stage1 pmnist: スモークの smoke_summary.json の全走が COMPLETE・checks_passed。
- stage1 scr / rlmnist: 加えて段 1 の箱 B の G1-PM（錨 9 走の provenance が g1_pm.pass_ で n_compared = 2040）。
  scr はスモークの G1-SCR・その対照・rss_probe が pass、rlmnist は G1-RL のスモーク（GPU）と選んだ device の決定性が pass。
- 本走（stage1・stage2・scr-lr0p005）は push 済みの 1 つの commit から回す: src/ configs/ specs/ に変更も未追跡も無く、
  HEAD が origin のどれかのブランチに含まれること（``require_committed_code``・dry-run は除く）。
- 完了の判定（``*_state``）: 出力のファイルがあるだけでは飛ばさない。provenance の地平線・状態・checks_passed・git_hash を
  現在の HEAD と照合し、合わなければ段を拒否する（黙って飛ばさない・古いコードの出力を再利用しない）。

インタプリタ（§2.1・参照を作ったものに固定）: 箱 B・RL は ``/usr/bin/python3``、SCR は ``proj_004_drift/.venv/bin/python``。
G0 で両方の ``sys.executable``・torch・numpy を assert し、SCR の検査の部分集合は .venv で回す。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'act_chimera_0913'
OUT = ROOT / 'results' / RUN_ID
SMOKE = ROOT / 'results' / '_smoke_act_chimera_0913'
LAUNCH = OUT / 'launch'                                         # 本走の記録
SMOKE_LAUNCH = SMOKE / 'launch'                                 # スモークの記録（本走の結果に混ぜない）
CHECKS = OUT / 'checks'
ACTIVE_LOCK = ROOT / 'results' / '.act_chimera_0913_active.lock'
PY_B = '/usr/bin/python3'                                       # 箱 B・RL（§2.1）
PY_SCR = '/home/issan/Projects/claude/proj_004_drift/.venv/bin/python'   # SCR（§2.1）
EXPECTED = {PY_B: dict(torch='2.13.0+cu130', numpy='2.5.2'), PY_SCR: dict(torch='2.13.0+cu130', numpy='2.5.1')}
RSS_HEADROOM = 1.5                                              # edge_law_0905.py:85
CAP = {'pmnist': 6, 'scr': 12, 'rlmnist': 8}                    # §7 の上限（RL は処理量で下げる）
REF_RSS_GIB = {'pmnist': 0.914, 'scr': 0.771, 'rlmnist': 1.6}   # §7 の参考実測（スモークの起動にだけ使う・出所を記録）
SEEDS = (0, 1, 2)
PM_TASKS, RL_TASKS = 120, 50
STAGE1_PM_ARMS = ('LR', 'ELU1', 'SMAXH', 'SMINH', 'VMIN', 'VMAX', 'SMAXS', 'SMINS', 'GN', 'FN', 'GD', 'FD', 'LR02',
                  'LRtw0', 'LRtw1', 'LRtw2', 'LRtw3')
PM_ANCHORS = ('LR', 'ELU1', 'SMAXH')
STAGE1_RL_ARMS = ('LR', 'ELU1', 'SMAXH', 'SMINH', 'LRtw0')
STAGE2_RL_ARMS = ('VMIN', 'VMAX', 'SMAXS', 'SMINS', 'GN', 'FN')
STAGE2_PM_LRS = ('5e-4', '2e-3')
SCR_ARMS = ('chLR_1216', 'chE_1216', 'chSMAXH_1216', 'chSMINH_1216', 'chVMIN_1216', 'chVMAX_1216', 'chSMAXS_1216',
            'chSMINS_1216', 'chGN_1216', 'chFN_1216', 'chGD_1216', 'chFD_1216')
SCR_ARMS_LR0P005 = tuple(a.replace('_1216', '_lr0p005_1216') for a in SCR_ARMS)
SCR_TOTAL = {**{a: 5_000_000 for a in SCR_ARMS}, **{a: 10_000_000 for a in SCR_ARMS_LR0P005}}
SMOKE_STEPS_SCR = 30_000
EST_WALL = {'pmnist': 90., 'scr': 1751., 'rlmnist': 1670.}      # §7 の見積もり（警告の基準・上限にしない）
# §8 で commit する出力（backup --execute が動かしてはならない）
COMMIT_TARGETS = ('results/act_chimera_0913/pmnist/*_rows.csv', 'results/act_chimera_0913/pmnist/*_units.npz',
                  'results/act_chimera_0913/pmnist/*_provenance.json', 'results/act_chimera_0913/pmnist/*_divergence.json',
                  'results/act_chimera_0913/rlmnist/*/s*/per_task.csv', 'results/act_chimera_0913/rlmnist/*/s*/provenance.json',
                  'results/act_chimera_0913/rlmnist/*/s*/divergence.json', 'results/act_chimera_0913/rlmnist/*/per_task.csv',
                  'results/act_chimera_0913/scr/arm_status/*', 'results/act_chimera_0913/scr/provenance.json',
                  'results/act_chimera_0913/scr/s_null.json', 'results/act_chimera_0913/verdict.csv',
                  'results/act_chimera_0913/seed_contrasts.csv', 'results/act_chimera_0913/summary.md',
                  'results/act_chimera_0913/report_provenance.json', 'results/act_chimera_0913/checks/static_checks.json',
                  'src/pmnist_0905.py', 'src/pmnist_rlmnist_0906.py', 'results/pmnist_rlmnist_0906/*/*',
                  'analysis/pmnist_0905/verdict_rlmnist.py')


class StageRefused(SystemExit):
    """ゲート・完了の照合・commit の条件が満たされないので段を起動しない（§6・§7）。"""


# ------------------------------------------------------------------ 機械の状態
def meminfo():
    info = {}
    with open('/proc/meminfo', encoding='utf-8') as fh:
        for line in fh:
            k, v = line.split(':', 1)
            info[k.strip()] = float(v.strip().split()[0]) / (1024. ** 2)     # GiB
    return info


def mem_available_gib():
    return meminfo()['MemAvailable']


def compute_J(rss_peak_gib, cap):
    """J = ⌊MemAvailable/(1.5·RSS_peak)⌋ − 1、上限 cap、下限 0（§7）。返り値 (J, MemAvailable)。"""
    avail = mem_available_gib()
    if rss_peak_gib <= 0:
        raise ValueError('rss_peak_gib must be positive (measure it in the smoke; no silent default)')
    J = int(avail // (RSS_HEADROOM * rss_peak_gib)) - 1
    return max(0, min(int(cap), J)), avail


def snapshot_processes():
    """他セッションの学習プロセス（ps）と nvidia-smi の compute プロセス（§7）。"""
    try:
        ps = subprocess.run(['ps', '-eo', 'pid,ppid,etimes,rss,args'], capture_output=True, text=True, timeout=20).stdout
        ps_lines = [l for l in ps.splitlines()[1:] if any(k in l for k in ('python', 'torch', '-m src.')) and 'ps -eo' not in l]
    except (OSError, subprocess.SubprocessError) as e:
        ps_lines = [f'ps failed: {e}']
    try:
        nv = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
                            capture_output=True, text=True, timeout=20)
        nv_out = nv.stdout.strip() if nv.returncode == 0 else f'nvidia-smi rc={nv.returncode}: {nv.stderr.strip()[:200]}'
    except (OSError, subprocess.SubprocessError) as e:
        nv_out = f'nvidia-smi failed: {e}'
    try:
        load = open('/proc/loadavg').read().split()[:3]
    except OSError:
        load = None
    return dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), ps=ps_lines, nvidia_smi=nv_out, meminfo=meminfo(), loadavg=load)


def _git(*a):
    try:
        return subprocess.check_output(['git', *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_head():
    return _git('rev-parse', 'HEAD') or 'unknown'


def machine():
    cpu = ''
    try:
        for line in open('/proc/cpuinfo', encoding='utf-8'):
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    gpu = ''
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return dict(hostname=socket.gethostname(), cpu=cpu, gpu=gpu)


def interpreter_info(py):
    code = 'import sys, torch, numpy; print(sys.executable); print(torch.__version__); print(numpy.__version__)'
    r = subprocess.run([py, '-c', code], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f'{py}: {r.stderr[-400:]}')
    ex, tv, nv = r.stdout.strip().splitlines()
    return dict(executable=ex, torch=tv, numpy=nv)


def assert_interpreters():
    out = {}
    for py, want in EXPECTED.items():
        info = interpreter_info(py)
        assert info['torch'] == want['torch'] and info['numpy'] == want['numpy'], (py, info, want)
        out[py] = info
    return out


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding='utf-8')


def _load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def launch_dir(stage):
    return SMOKE_LAUNCH if str(stage).startswith('smoke') else LAUNCH


# ------------------------------------------------------------------ 1 つの launcher だけ（段は同時に走らせない）
@contextlib.contextmanager
def launcher_lock(cmd):
    """サブコマンド全体で ``ACTIVE_LOCK`` を O_EXCL で取る。既存のロックは、launcher の PID か子の PID のどれかが
    生きていれば（段の名前に依らず・同じ段でも）拒否し、全部死んでいれば古いロックとして取り直す。"""
    ACTIVE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(str(ACTIVE_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                info = json.loads(ACTIVE_LOCK.read_text())
            except (OSError, ValueError):
                info = {}
            alive = [p for p in [info.get('launcher_pid')] + list(info.get('child_pids', [])) if _pid_alive(p)]
            if alive or attempt == 1:
                raise StageRefused(f"another launcher is active ({info.get('cmd')}, live PIDs {alive}, lock {ACTIVE_LOCK}); "
                                   f"stages never overlap (§7)")
            ACTIVE_LOCK.unlink()
    with os.fdopen(fd, 'w') as fh:
        fh.write(json.dumps(dict(cmd=cmd, launcher_pid=os.getpid(), child_pids=[], started=time.time())))
    try:
        yield ACTIVE_LOCK
    finally:
        try:
            ACTIVE_LOCK.unlink()
        except OSError:
            pass


def _lock_children(pids):
    try:
        info = json.loads(ACTIVE_LOCK.read_text())
    except (OSError, ValueError):
        return
    if info.get('launcher_pid') != os.getpid():
        return
    info['child_pids'] = [int(p) for p in pids if p]
    ACTIVE_LOCK.write_text(json.dumps(info))


# ------------------------------------------------------------------ ジョブキュー（§7）
class Job:
    def __init__(self, name, cmd, env=None, log=None, expect_wall=None, state=None):
        self.name, self.cmd, self.env, self.log = name, list(cmd), dict(env or {}), log
        self.expect_wall = expect_wall
        self.state = state                  # callable -> 'LAUNCH' | 'DONE'（完了の照合・合わなければ StageRefused）
        self.proc = self.pid = self.child_pid = self.pgid = self.start = self.end = self.rc = self.peak_kib = None
        self.time_file = None

    def record(self):
        return dict(name=self.name, cmd=self.cmd, pid=self.pid, child_pid=self.child_pid, pgid=self.pgid,
                    start=self.start, end=self.end, returncode=self.rc,
                    wall_seconds=(self.end - self.start) if (self.start and self.end) else None, peak_rss_kib=self.peak_kib,
                    status=('COMPLETE' if self.rc == 0 else ('FAILED' if self.rc is not None else ('RUNNING' if self.pid else 'PENDING'))),
                    log=str(self.log) if self.log else None)


def _parse_time_v(path):
    try:
        for line in Path(path).read_text(errors='replace').splitlines():
            if 'Maximum resident set size' in line:
                return float(line.rsplit(':', 1)[1].strip())
    except OSError:
        pass
    return None


def _children_of(pid):
    """/proc/<pid>/task/<pid>/children（無ければ /proc/*/stat の ppid を走査）。"""
    try:
        return [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except (OSError, ValueError):
        pass
    kids = []
    for d in Path('/proc').iterdir():
        if d.name.isdigit():
            try:
                if int((d / 'stat').read_text().rsplit(')', 1)[1].split()[1]) == int(pid):
                    kids.append(int(d.name))
            except (OSError, ValueError, IndexError):
                continue
    return kids


def run_queue(jobs, stage, rss_peak_gib, cap, poll=5., dry_run=False, env_name='', rss_source='', stop_on_failure=True):
    """ジョブを J の枠で回す。J は**起動直前に毎回**計算し直す。返り値は status dict（<launch>/<stage>_status.json にも書く）。
    完了の照合（Job.state）はキューを組む時に全ジョブで行い、合わないものがあれば何も起動せずに StageRefused。"""
    ldir = launch_dir(stage)
    status_path = ldir / f'{stage}_status.json'
    prov_path = ldir / f'{stage}_provenance.json'
    states = {j.name: (j.state() if j.state else 'LAUNCH') for j in jobs}      # StageRefused はここで上がる
    pending = [j for j in jobs if states[j.name] == 'LAUNCH']
    skipped = [j.name for j in jobs if states[j.name] == 'DONE']
    st = dict(stage=stage, env=env_name, started=time.time(), git_head=git_head(), machine=machine(), rss_peak_gib=rss_peak_gib,
              rss_source=rss_source, cap=cap, headroom=RSS_HEADROOM, jobs_total=len(jobs), skipped_complete=skipped,
              J_history=[], events=[], jobs=[j.record() for j in jobs], dry_run=dry_run, stopped_after_failure=None)
    if dry_run:
        J, avail = compute_J(rss_peak_gib, cap)
        st['J_history'].append(dict(time=time.time(), J=J, mem_available_gib=avail, running=0))
        st['plan'] = [j.record() for j in pending]
        print(json.dumps(dict(stage=stage, J=J, mem_available_gib=avail, n_pending=len(pending), n_skipped=len(skipped),
                              rss_peak_gib=rss_peak_gib, rss_source=rss_source), indent=1))
        for j in pending:
            print('  ', ' '.join(j.cmd))
        return st
    before = snapshot_processes()
    _dump(prov_path, dict(stage=stage, before=before, git_head=st['git_head'], machine=st['machine']))
    running = []
    warned = set()
    t_stage = time.time()
    try:
        while pending or running:
            # 完了の回収（/usr/bin/time が終わっても子の学習プロセスが生きていれば枠を空けない）
            for j in list(running):
                if j.child_pid is None and j.proc.poll() is None:
                    kids = _children_of(j.pid)
                    j.child_pid = kids[0] if kids else None
                rc = j.proc.poll()
                if rc is not None and not (j.child_pid and _pid_alive(j.child_pid)):
                    j.rc, j.end = rc, time.time()
                    j.peak_kib = _parse_time_v(j.time_file)
                    running.remove(j)
                    st['events'].append(dict(time=j.end, event='exit', name=j.name, pid=j.pid, child_pid=j.child_pid, rc=rc,
                                             wall=j.end - j.start, peak_kib=j.peak_kib))
                    print(f"[{time.strftime('%H:%M:%S')}] exit {j.name} rc={rc} wall={j.end - j.start:.0f}s peak={j.peak_kib}", flush=True)
                    if rc != 0 and stop_on_failure and pending:
                        st['stopped_after_failure'] = dict(job=j.name, rc=rc, not_launched=[p.name for p in pending])
                        print(f'STOP: {j.name} failed (rc={rc}); {len(pending)} pending jobs are not launched (§7)', flush=True)
                        pending = []
                elif rc is not None:
                    if j.name not in warned:
                        warned.add(j.name)
                        st['events'].append(dict(time=time.time(), event='orphan_child_alive', name=j.name, child_pid=j.child_pid))
                elif j.expect_wall and (time.time() - j.start) > 3 * j.expect_wall and j.name not in warned:
                    warned.add(j.name)
                    st['events'].append(dict(time=time.time(), event='slow_warning', name=j.name, pid=j.pid, elapsed=time.time() - j.start))
                    print(f'WARNING {j.name} exceeds 3x the estimate ({j.expect_wall:.0f}s); not killed (§7)', flush=True)
            # 起動（直前に J を計算し直す）
            launched = False
            if pending:
                J, avail = compute_J(rss_peak_gib, cap)
                st['J_history'].append(dict(time=time.time(), J=J, mem_available_gib=avail, running=len(running)))
                if len(running) < J and avail >= RSS_HEADROOM * rss_peak_gib:
                    j = pending.pop(0)
                    ldir.mkdir(parents=True, exist_ok=True)
                    j.time_file = ldir / f'{stage}_{j.name}.time'
                    j.log = j.log or ldir / f'{stage}_{j.name}.log'
                    cmd = ['/usr/bin/time', '-v', '-o', str(j.time_file)] + j.cmd
                    env = dict(os.environ, PYTHONPATH=str(ROOT), **j.env)
                    with open(j.log, 'ab') as fh:
                        j.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                                  start_new_session=True)
                    j.pid, j.start = j.proc.pid, time.time()
                    j.pgid = j.pid                                   # start_new_session: time が新しいグループの長
                    for _ in range(20):
                        kids = _children_of(j.pid)
                        if kids or j.proc.poll() is not None:
                            j.child_pid = kids[0] if kids else None
                            break
                        time.sleep(0.05)
                    running.append(j)
                    launched = True
                    st['events'].append(dict(time=j.start, event='launch', name=j.name, pid=j.pid, child_pid=j.child_pid,
                                             pgid=j.pgid, cmd=cmd, J=J, mem_available_gib=avail))
                    print(f"[{time.strftime('%H:%M:%S')}] launch {j.name} pid={j.pid} child={j.child_pid} pgid={j.pgid} "
                          f"(running {len(running)}/J={J}, avail {avail:.1f} GiB)", flush=True)
            st['jobs'] = [j.record() for j in jobs]
            _dump(status_path, st)
            _lock_children([p for j in running for p in (j.pid, j.child_pid)])
            if not launched:
                time.sleep(poll if running else min(poll, 2.))
                if pending and not running:
                    J, avail = compute_J(rss_peak_gib, cap)
                    if J < 1 or avail < RSS_HEADROOM * rss_peak_gib:
                        print(f'waiting: MemAvailable {avail:.2f} GiB < {RSS_HEADROOM * rss_peak_gib:.2f} GiB or J={J}; nothing launched', flush=True)
                        time.sleep(30)
    finally:
        st['finished'] = time.time()
        st['wall_seconds'] = time.time() - t_stage
        st['jobs'] = [j.record() for j in jobs]
        _dump(status_path, st)
        after = snapshot_processes()
        _dump(prov_path, dict(stage=stage, before=before, after=after, git_head=st['git_head'], machine=st['machine'],
                              wall_seconds=st['wall_seconds'], jobs=st['jobs'], J_history=st['J_history'],
                              stopped_after_failure=st['stopped_after_failure']))
        _lock_children([])
    return st


def queue_ok(st):
    """待ち行列の全ジョブが rc 0（飛ばしたものは完了の照合済み）。"""
    return st.get('stopped_after_failure') is None and all(j['returncode'] in (0, None) and j['status'] != 'FAILED'
                                                          for j in st['jobs'] if j['name'] not in st['skipped_complete'])


# ------------------------------------------------------------------ 完了の照合（出力があるだけでは飛ばさない）
def pm_state(outdir, arm, seed, tasks=PM_TASKS, lr=None):
    p = Path(outdir) / f'{arm}_s{seed}_provenance.json'
    prov = _load(p)
    if prov is None:
        return 'LAUNCH'
    why = []
    if int(prov.get('tasks', -1)) != tasks:
        why.append(f"tasks={prov.get('tasks')} != {tasks}")
    if prov.get('status') not in ('COMPLETE', 'DIVERGED'):
        why.append(f"status={prov.get('status')}")
    if prov.get('checks_passed') is not True:
        why.append(f"checks_passed={prov.get('checks_passed')} failed={prov.get('failed_checks')}")
    if lr is not None and float(prov.get('lr', -1)) != float(lr):
        why.append(f"lr={prov.get('lr')} != {lr}")
    if prov.get('git_hash') != git_head():
        why.append(f"git_hash={prov.get('git_hash')} != HEAD {git_head()}")
    if why:
        raise StageRefused(f'{p}: existing output does not match this stage ({"; ".join(why)}); move it away first (§6)')
    return 'DONE'


def _device_type(d):
    return None if d is None else str(d).split(':')[0]


def rl_state(outdir, arm, seed, tasks=RL_TASKS, device=None):
    """``device`` を渡すと、provenance の ``machine.device`` が違う走も拒否する（追補 2-3: RL は全腕を 1 つの device で回す。
    GPU の走を CPU の段 1 が DONE として飛ばすと、LR と双子の device が混ざる）。"""
    p = Path(outdir) / arm / f's{seed}' / 'provenance.json'
    prov = _load(p)
    if prov is None:
        return 'LAUNCH'
    why = []
    if int(prov.get('n_tasks', -1)) != tasks:
        why.append(f"n_tasks={prov.get('n_tasks')} != {tasks}")
    run_dev = _device_type((prov.get('machine') or {}).get('device'))
    if device is not None and run_dev != _device_type(device):
        why.append(f'machine.device={run_dev} != --device {device}')
    if prov.get('status') not in ('COMPLETE', 'INCOMPLETE'):
        why.append(f"status={prov.get('status')}")
    if prov.get('checks_passed') is False:
        why.append(f"checks_passed=False failed={prov.get('failed_checks')}")
    if prov.get('git_hash') != git_head():
        why.append(f"git_hash={prov.get('git_hash')} != HEAD {git_head()}")
    if why:
        raise StageRefused(f'{p}: existing output does not match this stage ({"; ".join(why)}); move it away first (§6)')
    return 'DONE'


def scr_state(outdir, arm):
    p = Path(outdir) / 'arm_status' / f'{arm}_done.json'
    done = _load(p)
    div = _load(Path(outdir) / 'arm_status' / f'{arm}.json')
    if done is None and div is None:
        return 'LAUNCH'
    why = []
    rec = done or div
    if done is not None and int(done.get('total_steps', -1)) != SCR_TOTAL[arm]:
        why.append(f"total_steps={done.get('total_steps')} != {SCR_TOTAL[arm]}")
    head = rec.get('git_head') or rec.get('git_hash')
    if head != git_head():
        why.append(f'git_head={head} != HEAD {git_head()}')
    if why:
        raise StageRefused(f'{p}: existing output does not match this stage ({"; ".join(why)}); move it away first (§6)')
    return 'DONE'


# ------------------------------------------------------------------ 本走の前提（commit・ゲート）
def require_committed_code():
    """§6: 本走は push した 1 つの commit から回す。src/ configs/ specs/ に変更も未追跡も無く、HEAD が origin に含まれる。"""
    dirty = _git('status', '--porcelain', '--untracked-files=all', '--', 'src', 'configs', 'specs')
    dirty = [l for l in (dirty or '').splitlines() if '__pycache__' not in l]
    on_remote = _git('branch', '-r', '--contains', 'HEAD')
    rec = dict(git_head=git_head(), dirty_src=dirty, on_remote=on_remote, pass_=bool(not dirty and on_remote))
    if not rec['pass_']:
        raise StageRefused(f'main runs need the committed and pushed code (§6): dirty={dirty[:10]} on_remote={on_remote!r}')
    return rec


def code_sha256_now():
    """G0 が記録し、ゲートが照合するコードの sha256（2-4 A7・S6）: 共有モジュール ``act_chimera_0913.py``・
    ``act_chimera_*_0913.py``（3 ランナー・判定器・launcher）・2 つのテストファイル。"""
    src = ROOT / 'src'
    paths = {*src.glob('act_chimera_*_0913.py'), src / 'act_chimera_0913.py', src / 'test_act_chimera_0913.py',
             src / 'test_act_chimera_report_0913.py'}
    return {p.name: _sha(p) for p in sorted(paths)}


def _static_checks_pass():
    """static_checks.json の G0（両インタプリタ）と G0.5 が pass で、**いまのコードと HEAD に対して**記録されたこと
    （2-4 A7）。``pass_`` だけを見ると、追補より前のコードで回した G0・G0.5 が通ってしまう。返り値 (bool の dict, 詳細)。"""
    rec = _load(CHECKS / 'static_checks.json') or {}
    g0, g05 = rec.get('_G0') or {}, rec.get('G05') or {}
    now, got = code_sha256_now(), g0.get('code_sha256') or {}
    stale = sorted(k for k in set(now) | set(got) if now.get(k) != got.get(k))
    head = git_head()
    ok = dict(G0=bool(g0.get('pass_')), G0_scr=bool((rec.get('_G0_scr') or {}).get('pass_')), G05=bool(g05.get('pass_')),
              G0_code_current=not stale,
              G0_git_head_current=bool(head is not None and g0.get('git_head') == head),
              G05_report_current=bool(g05.get('report_sha256') is not None
                                      and g05.get('report_sha256') == now.get('act_chimera_report_0913.py')))
    return ok, dict(stale_code_sha256=stale, g0_git_head=g0.get('git_head'), head=head,
                    g05_report_sha256=g05.get('report_sha256'))


def g1_pm_state(outdir=OUT / 'pmnist'):
    """段 1 の箱 B の G1-PM: 錨 9 走の provenance が g1_pm.pass_ で n_compared = 17 × 120 = 2040。"""
    rows = {}
    for arm in PM_ANCHORS:
        for s in SEEDS:
            prov = _load(Path(outdir) / f'{arm}_s{s}_provenance.json') or {}
            g = (prov.get('checks') or {}).get('g1_pm') or {}
            rows[f'{arm}_s{s}'] = dict(pass_=bool(g.get('pass_')), n_compared=g.get('n_compared'),
                                       git_hash=prov.get('git_hash'))
    ok = all(v['pass_'] and v['n_compared'] == 17 * PM_TASKS and v['git_hash'] == git_head() for v in rows.values())
    return dict(pass_=ok, anchors=rows)


def require_gates(stage, env, device=None):
    """§6・§7 の停止規則。満たさなければ StageRefused。返り値は provenance に書く記録。"""
    rec = dict(stage=stage, env=env)
    rec['static'], rec['static_detail'] = _static_checks_pass()
    miss = [k for k, v in rec['static'].items() if not v]
    if miss:
        raise StageRefused(f'{stage} {env}: static checks not passed or stale: {miss} {rec["static_detail"]} '
                           '(re-run g0 and g05 on the committed HEAD; §6・2-4 S6/A7)')
    smoke = _load(SMOKE / 'pmnist' / 'smoke_summary.json')
    rec['smoke_pmnist'] = bool(smoke and smoke.get('pass_'))
    if not rec['smoke_pmnist']:
        raise StageRefused(f'{stage} {env}: box B smoke not passed (smoke_summary.json pass_; §6)')
    # 追補 2-2: 双子の摂動を変えたので、それより前のコードのスモーク（W1[k, 0] の 1 ulp・S18b の記録なし）では通さない
    tw = (_load(SMOKE / 'pmnist' / 'LRtw0_s0_provenance.json') or {}).get('checks') or {}
    rec['smoke_pmnist_twin_addendum2'] = bool((tw.get('s18') or {}).get('registered_delta') == 1e-6
                                              and (tw.get('s18') or {}).get('pass_') and 's18b' in tw)
    if not rec['smoke_pmnist_twin_addendum2']:
        raise StageRefused(f'{stage} {env}: box B smoke predates addendum 2 (LRtw0_s0 has no b1[k] +1e-6 S18 / S18b record); '
                           're-run `smoke --env pmnist`')
    if env in ('scr', 'rlmnist') or stage == 'stage2':
        rec['g1_pm'] = g1_pm_state()
        if not rec['g1_pm']['pass_']:
            raise StageRefused(f'{stage} {env}: G1-PM not passed on stage-1 box B; SCR and RL wait until the cause is known (§7)')
    if env == 'scr':
        for name in ('g1_scr', 'g1_scr_control', 'rss_probe'):
            r = _load(SMOKE / 'scr' / 'sanity' / f'{name}.json')
            rec[name] = bool(r and r.get('pass_'))
            if not rec[name]:
                raise StageRefused(f'{stage} scr: smoke {name} not passed (§6)')
    if env == 'rlmnist':
        g1 = _load(SMOKE / 'rlmnist' / 'g1' / 'g1_smoke.json')
        # 追補 2-1・2-2: 対照は b1[0] +1e−3（G1-RL）と、b1[0] を除いた sha256 の双子（決定性）。旧コードの記録
        # （W1[0, 0] の対照・全 params の sha256 の双子）は、pass_ が True でも通さない
        rec['g1_rl_smoke'] = bool(g1 and g1.get('pass_') and (g1.get('control_b1_0_plus_1e-3') or {}).get('fails_as_required'))
        det = _load(SMOKE / 'rlmnist' / f'determinism_{device}' / 'determinism.json') if device else None
        rec['determinism'] = bool(det and det.get('pass_') and (det.get('control_twin_b1_0') or {}).get('fails_as_required'))
        if not rec['g1_rl_smoke']:
            raise StageRefused(f'{stage} rlmnist: G1-RL smoke (GPU) not passed; RL is not launched (§2.3・§7)')
        if not rec['determinism']:
            raise StageRefused(f'{stage} rlmnist: new-side determinism on {device} not passed (§2.3)')
    return rec


# ------------------------------------------------------------------ RSS_peak の実測値（スモークから）
def _rl_job_device(name):
    """RL のスモークのジョブ名から device（``rl_{arm}_s{seed}_{device}``・``rl_determinism_{device}``・GPU だけの
    ``rl_g1_smoke``）。分からなければ None（数えない）。"""
    if name == 'rl_g1_smoke':
        return 'cuda'
    for d in ('cpu', 'cuda'):
        if name.endswith(f'_{d}'):
            return d
    return None


def rss_peak_from_smoke(env, horizon=None, device=None):
    """スモークの実測から RSS_peak [GiB]。無ければ (None, None)（黙って既定値に落ちない）。
    SCR は rss_probe の地平線ごとの外挿（本走 5M・退避枝 10M）から ``horizon`` の値を取る。
    RL は ``device`` の実測だけを使う（追補 2-3: CPU の本走は CPU の 1.02 GiB で J を決め、GPU の走の host RSS
    1.73 GiB を混ぜない）。provenance は ``machine.device`` が一致するもの、/usr/bin/time はジョブ名の device が一致するもの。"""
    if env == 'pmnist':
        d = _load(SMOKE / 'pmnist' / 'smoke_summary.json')
        if d and d.get('pass_'):
            vals = [r.get('peak_rss_kib') for r in d.get('runs', []) if r.get('peak_rss_kib')]
            st = _load(SMOKE_LAUNCH / 'smoke-pmnist_status.json') or {}
            vals += [j['peak_rss_kib'] for j in st.get('jobs', []) if j.get('peak_rss_kib')]
            if vals:
                return max(vals) / 1024. ** 2, str(SMOKE / 'pmnist' / 'smoke_summary.json')
        return None, None
    if env == 'scr':
        d = _load(SMOKE / 'scr' / 'sanity' / 'rss_probe.json')
        if d and d.get('pass_'):
            proj = d.get('projected_peak_gib') or {}
            v = proj.get(str(int(horizon))) if horizon else d.get('worst_peak_gib')
            if v:
                return float(v), f"rss_probe projected_peak_gib[{horizon}]"
        return None, None
    if env == 'rlmnist':
        dev = _device_type(device)
        if dev not in ('cpu', 'cuda'):
            raise ValueError(f'rss_peak_from_smoke(rlmnist) needs device cpu|cuda, got {device!r} (addendum 2-3)')
        vals = []
        for q in (SMOKE / 'rlmnist').rglob('provenance.json'):
            pr = _load(q) or {}
            v = pr.get('peak_rss_kib')
            if v and _device_type((pr.get('machine') or {}).get('device')) == dev:
                vals.append(float(v))
        for stp in SMOKE_LAUNCH.glob('smoke-rlmnist-*_status.json'):
            vals += [j['peak_rss_kib'] for j in (_load(stp) or {}).get('jobs', [])
                     if j.get('peak_rss_kib') and _rl_job_device(j.get('name', '')) == dev]
        src = (f'{dev.upper()} smoke measurement (追補 2-3): smoke provenance with machine.device == {dev} '
               f'+ smoke-rlmnist-* /usr/bin/time of *_{dev} jobs')
        return (max(vals) / 1024. ** 2, src) if vals else (None, None)
    raise KeyError(env)


def resolve_rss(env, override, horizon=None, device=None):
    if override:
        return float(override), 'override (--rss-peak-gib)'
    v, src = rss_peak_from_smoke(env, horizon, device=device)
    if v is None:
        raise StageRefused(f'{env}: no measured RSS_peak from a passed smoke; run the smoke first or pass --rss-peak-gib explicitly (§7)')
    return v, src


def smoke_rss(env, override):
    """スモークの起動に使う RSS: 明示の値、無ければ §7 の参考実測（出所を status に書く）。"""
    if override:
        return float(override), 'override (--rss-peak-gib)'
    return REF_RSS_GIB[env], f'spec §7 reference measurement ({REF_RSS_GIB[env]} GiB)'


# ------------------------------------------------------------------ ジョブの組み立て
def pm_job(arm, seed, outdir, lr=None, tasks=None, extra=()):
    cmd = [PY_B, '-m', 'src.act_chimera_pmnist_0913', '--arm', arm, '--seed', str(seed), '--outdir', str(outdir)]
    if lr is not None:
        cmd += ['--lr', str(lr)]
    if tasks is not None:
        cmd += ['--tasks', str(tasks)]
    cmd += list(extra)
    tag = f'{arm}_s{seed}'
    return Job(f'pm_{tag}' + (f'_lr{lr}' if lr else ''), cmd, env={'OMP_NUM_THREADS': '1'}, expect_wall=EST_WALL['pmnist'],
               state=lambda: pm_state(outdir, arm, seed, tasks or PM_TASKS, lr))


def scr_job(arm, outdir, steps=None, extra=()):
    cmd = [PY_SCR, '-m', 'src.act_chimera_scr_0913', '--arm', arm, '--outdir', str(outdir)]
    if steps is not None:
        cmd += ['--steps', str(steps)]
    cmd += list(extra)
    return Job(f'scr_{arm}', cmd, env={'OMP_NUM_THREADS': '1'}, expect_wall=EST_WALL['scr'] * (2 if 'lr0p005' in arm else 1),
               state=(lambda: scr_state(outdir, arm)) if steps is None else None)


def rl_job(arm, seed, outdir, device, tasks=None, extra=()):
    cmd = [PY_B, '-m', 'src.act_chimera_rlmnist_0913', '--arm', arm, '--seed', str(seed), '--device', device, '--outdir', str(outdir)]
    if tasks is not None:
        cmd += ['--tasks', str(tasks)]
    cmd += list(extra)
    return Job(f'rl_{arm}_s{seed}_{device}', cmd, env={'OMP_NUM_THREADS': '1'}, expect_wall=EST_WALL['rlmnist'],
               state=(lambda: rl_state(outdir, arm, seed, device=device)) if tasks is None else None)


def post_job(name, py, args):
    return Job(name, [py, '-m'] + list(args), env={'OMP_NUM_THREADS': '1'})


def _fresh_smoke_dir(d):
    d = Path(d)
    assert SMOKE in d.resolve().parents or d.resolve() == SMOKE, d
    if d.exists():
        shutil.rmtree(d)
    return d


# ------------------------------------------------------------------ G0・スモーク・G0.5
def g0(argv):
    """G0: S 検査（test_act_chimera_0913.collect）を /usr/bin/python3 で走らせ、SCR の部分集合を .venv でも走らせて、
    checks/static_checks.json に写す。"""
    CHECKS.mkdir(parents=True, exist_ok=True)
    interp = assert_interpreters()
    t0 = time.time()
    rec = None
    for key, py, arg in (('_G0', PY_B, []), ('_G0_scr', PY_SCR, ['--scr'])):
        r = subprocess.run([py, '-m', 'src.test_act_chimera_0913'] + arg, cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1'))
        log = CHECKS / f'static_checks{key}.log'
        log.write_text(r.stdout[-200000:] + '\n--- stderr ---\n' + r.stderr[-200000:], encoding='utf-8')
        if r.returncode != 0:
            prev = rec or {}
            prev[key] = dict(pass_=False, returncode=r.returncode, interpreter=py, stderr_tail=r.stderr[-4000:])
            _dump(CHECKS / 'static_checks.json', prev)
            raise SystemExit(f'G0 FAILED under {py} (rc={r.returncode}); see {log}')
        sub = json.loads(r.stdout[r.stdout.index('{'):])
        if rec is None:
            rec = sub
        else:
            rec['_scr_interpreter_records'] = sub
        rec[key] = dict(pass_=True, interpreter=py, env=sub.get('_env'), wall_seconds=time.time() - t0)
    rec['_G0'].update(interpreters=interp, git_head=git_head(), machine=machine(),
                      code_sha256=code_sha256_now(),
                      spec_sha256=_sha(ROOT / 'specs' / 'spec_act_chimera_0913.md'))
    _dump(CHECKS / 'static_checks.json', rec)
    print(f'G0 PASS ({time.time() - t0:.0f}s) -> {CHECKS / "static_checks.json"}')


def smoke(args):
    """スモーク（§6）。箱 B → SCR → RL の順・同時にしない。各環境のピーク RSS を実測する。"""
    env = args.env
    if env == 'pmnist':
        out = SMOKE / 'pmnist'
        out.mkdir(parents=True, exist_ok=True)
        rss, src = smoke_rss('pmnist', args.rss_peak_gib)
        j = Job('pm_smoke', [PY_B, '-m', 'src.act_chimera_pmnist_0913', '--smoke', '--outdir', str(out)], env={'OMP_NUM_THREADS': '1'},
                expect_wall=17 * 3 * 15.)
        return run_queue([j], 'smoke-pmnist', rss_peak_gib=rss, cap=1, dry_run=args.dry_run, env_name=env, rss_source=src)
    if env == 'scr':
        rss, src = smoke_rss('scr', args.rss_peak_gib)
        out, off1 = SMOKE / 'scr', SMOKE / 'scr_offset1'
        if not args.dry_run:                          # 前回のスモークのログを再開しない（コードを変えた後の再スモーク）
            _fresh_smoke_dir(out)
            _fresh_smoke_dir(off1)
        jobs = [scr_job(a, out, steps=SMOKE_STEPS_SCR) for a in SCR_ARMS]
        st = run_queue(jobs, 'smoke-scr', rss_peak_gib=rss, cap=CAP['scr'], dry_run=args.dry_run, env_name=env, rss_source=src)
        if args.dry_run or not queue_ok(st):
            return st
        post = [post_job('scr_g1', PY_SCR, ['src.act_chimera_scr_0913', '--g1', '--smoke', '--outdir', str(out)]),
                post_job('scr_offset1', PY_SCR, ['src.act_chimera_scr_0913', '--arm', 'chLR_1216', '--steps', str(SMOKE_STEPS_SCR),
                                                 '--generator-offset', '1', '--outdir', str(off1)]),
                post_job('scr_g1_control', PY_SCR, ['src.act_chimera_scr_0913', '--g1-control', str(off1), '--outdir', str(out)]),
                post_job('scr_rss_probe', PY_SCR, ['src.act_chimera_scr_0913', '--rss-probe', '--outdir', str(out)])]
        post += [post_job(f'scr_readout_{a}', PY_SCR, ['src.act_chimera_scr_0913', '--readout', a, '--no-power-iter',
                                                     '--outdir', str(out)]) for a in SCR_ARMS]     # G0.5 のスキーマ照合用
        return run_queue(post, 'smoke-scr-post', rss_peak_gib=rss, cap=1, env_name=env, rss_source=src)
    if env == 'rlmnist':
        rss, src = smoke_rss('rlmnist', args.rss_peak_gib)
        out = SMOKE / 'rlmnist'
        dev = args.device or 'cpu'
        res = {}
        g1 = [post_job('rl_g1_smoke', PY_B, ['src.act_chimera_rlmnist_0913', '--g1-smoke', '--device', 'cuda', '--outdir', str(out)])]
        res['g1'] = run_queue(g1, 'smoke-rlmnist-g1', rss_peak_gib=rss, cap=1, dry_run=args.dry_run, env_name=env, rss_source=src)
        det = [post_job(f'rl_determinism_{d}', PY_B, ['src.act_chimera_rlmnist_0913', '--determinism', '--device', d,
                                                      '--outdir', str(out)]) for d in ('cpu', 'cuda')]
        res['determinism'] = run_queue(det, 'smoke-rlmnist-determinism', rss_peak_gib=rss, cap=1, dry_run=args.dry_run,
                                       env_name=env, rss_source=src, stop_on_failure=False)
        # H 家族と双子の 1 タスク（CPU と GPU の時間を測る・未決 5）
        timing = {}
        for d in ('cpu', 'cuda'):
            js = [rl_job(a, 0, out / f'timing_{d}', d, tasks=1) for a in STAGE1_RL_ARMS]
            st = run_queue(js, f'smoke-rlmnist-timing-{d}', rss_peak_gib=rss, cap=1, dry_run=args.dry_run, env_name=env, rss_source=src)
            timing[d] = {j['name']: j['wall_seconds'] for j in st['jobs']}
        # G0.5 が読む配置（rlmnist/{ARM}/s{seed}）: H 家族と双子 × seed 0–2 の 1 タスク（--device の device）
        g05_jobs = [rl_job(a, s, out, dev, tasks=1) for a in STAGE1_RL_ARMS for s in SEEDS]
        res['g05_layout'] = run_queue(g05_jobs, f'smoke-rlmnist-g05-{dev}', rss_peak_gib=rss, cap=CAP['rlmnist'],
                                      dry_run=args.dry_run, env_name=env, rss_source=src)
        if not args.dry_run:
            _dump(SMOKE_LAUNCH / 'smoke-rlmnist_timing.json', timing)
            thr = rl_throughput(out / 'throughput', dev, rss_peak_gib=rss)
            _dump(SMOKE_LAUNCH / f'smoke-rlmnist_throughput_{dev}.json', thr)
        return res
    raise SystemExit(f'unknown env {env}')


def rl_throughput(out, device, rss_peak_gib, parallels=(1, 2, 4, None)):
    """RL の並列数（§7）: 並列 p で 1 タスクの走を p 本回し、各プロセスの provenance の steps_per_s（タスク内の実時間から・
    起動と MNIST 読み込みを含まない）を足した処理量を測る。p 本が実際に同時に走ったこと（事象から数えた同時実行数の最大 = p）
    を確かめ、そうでなければ NOT_MEASURED。最良の 90% 以内に入る最小の並列数を選ぶ。"""
    res = {}
    for p in parallels:
        if p is None:
            p, _ = compute_J(rss_peak_gib, CAP['rlmnist'])
            if p in res or p < 1:
                continue
        pdir = out / f'p{p}'
        if pdir.exists():
            shutil.rmtree(pdir)
        jobs = [rl_job('LR', s, pdir, device, tasks=1) for s in range(p)]
        st = run_queue(jobs, f'smoke-rlmnist-thr{p}', rss_peak_gib=rss_peak_gib, cap=p, env_name='rlmnist')
        ev = sorted([(e['time'], +1) for e in st['events'] if e['event'] == 'launch'] +
                    [(e['time'], -1) for e in st['events'] if e['event'] == 'exit'])
        cur = mx = 0
        for _, dlt in ev:
            cur += dlt
            mx = max(mx, cur)
        sps = [(_load(pdir / 'LR' / f's{s}' / 'provenance.json') or {}).get('steps_per_s') for s in range(p)]
        ok = mx == p and all(v for v in sps) and queue_ok(st)
        res[p] = dict(parallel=p, max_concurrent=mx, steps_per_s_each=sps,
                      steps_per_s=float(sum(sps)) if ok else None, status='OK' if ok else 'NOT_MEASURED')
    good = {p: v for p, v in res.items() if v['steps_per_s']}
    if not good:
        return dict(measurements=res, chosen_parallel=None, rule='smallest parallel within 90% of the best', status='NOT_MEASURED')
    best = max(v['steps_per_s'] for v in good.values())
    chosen = min(p for p, v in good.items() if v['steps_per_s'] >= 0.9 * best)
    return dict(measurements=res, best_steps_per_s=best, chosen_parallel=chosen, device=device,
                rule='smallest parallel within 90% of the best (sum of per-process steps_per_s; concurrency verified)')


def g05(args):
    """G0.5（§6）: 判定器を 3 環境のスモーク出力に端から端まで掛け、全行が値を持つこと。3 環境とも少なくとも 1 つの腕が
    読めていること。変異対照: 環境ごとに読み出しのキーを 1 つ消すと SchemaError で落ちること。"""
    from src import act_chimera_report_0913 as RP
    import numpy as np
    CHECKS.mkdir(parents=True, exist_ok=True)
    root = Path(args.root or SMOKE)
    t0 = time.time()
    R = RP.main(['--root', str(root), '--out', str(root / 'report'), '--smoke'])
    n = RP.all_rows_have_values(R)
    loaded = {e: sorted({r['label'] for r in R.rows if r['env'] == e and r['item'] in ('STATUS', 'SCR_STATUS')})
              for e in ('pmnist', 'rlmnist', 'scr')}
    for e, labs in loaded.items():
        if not any(l == 'COMPLETE' for l in labs):
            raise SystemExit(f'G0.5 FAILED: no {e} arm was loaded from the smoke outputs (statuses {labs})')
    scratch_base = Path(args.scratch) if args.scratch else Path(os.environ.get('ACT_CHIMERA_SCRATCH', '/tmp'))
    controls = {}
    for e, pattern, key in (('pmnist', 'pmnist/*_readout.npz', 'depth_vel'), ('rlmnist', 'rlmnist/*/s0/readout_s0.npz', 'mob_band'),
                            ('scr', 'scr/readout_*.npz', 'lam_out')):
        scratch = scratch_base / f'g05_mut_{e}'
        if scratch.exists():
            shutil.rmtree(scratch)
        shutil.copytree(root, scratch, ignore=shutil.ignore_patterns('report', 'ckpts', 'launch', 'rss', 'throughput', 'timing_*',
                                                                     'determinism_*', 'g1', 'scr_offset1'))
        victims = sorted(scratch.glob(pattern))
        if not victims:
            raise SystemExit(f'G0.5 mutation control ({e}): no readout matching {pattern}')
        p = victims[0]
        with np.load(p, allow_pickle=True) as z:
            arrs = {k: z[k] for k in z.files if k != key}
        np.savez(p, **arrs)
        try:
            RP.main(['--root', str(scratch), '--out', str(scratch / 'report'), '--smoke'])
            raise SystemExit(f'G0.5 mutation control FAILED ({e}): deleting {key} did not make the report fail')
        except RP.SchemaError as exc:
            controls[e] = dict(deleted_key=key, file=str(p), error=str(exc))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    rec = _load(CHECKS / 'static_checks.json') or {}
    rec['G05'] = dict(pass_=True, n_rows=n, root=str(root), loaded_statuses=loaded, controls=controls,
                      report_sha256=_sha(ROOT / 'src' / 'act_chimera_report_0913.py'), git_head=git_head(),
                      wall_seconds=time.time() - t0)
    _dump(CHECKS / 'static_checks.json', rec)
    print(f'G0.5 PASS: {n} verdict rows; mutation controls raised SchemaError in all 3 environments ({time.time() - t0:.0f}s)')


# ------------------------------------------------------------------ 段 1・段 2・退避枝
def _stage_prov(stage, rec):
    _dump(LAUNCH / f'{stage}_preconditions.json', rec)


def stage1(args):
    env = args.env
    pre = {} if args.dry_run else dict(committed=require_committed_code(), gates=require_gates('stage1', env, args.device))
    if env == 'pmnist':
        rss, src = resolve_rss('pmnist', args.rss_peak_gib)
        out = OUT / 'pmnist'
        jobs = [pm_job(a, s, out) for a in STAGE1_PM_ARMS for s in SEEDS]
        if not args.dry_run:
            _stage_prov('stage1-pmnist', pre)
        st = run_queue(jobs, 'stage1-pmnist', rss, CAP['pmnist'], dry_run=args.dry_run, env_name=env, rss_source=src)
        if not args.dry_run:
            st['g1_pm'] = g1_pm_state(out)
            print('G1-PM:', json.dumps(st['g1_pm']['pass_']), flush=True)
        return st
    if env == 'scr':
        rss, src = resolve_rss('scr', args.rss_peak_gib, horizon=5_000_000)
        out = OUT / 'scr'
        jobs = [scr_job(a, out) for a in SCR_ARMS]
        if not args.dry_run:
            _stage_prov('stage1-scr', pre)
        st = run_queue(jobs, 'stage1-scr', rss, CAP['scr'], dry_run=args.dry_run, env_name=env, rss_source=src)
        if not args.dry_run and queue_ok(st):
            # provenance（G1-SCR の記録を含む・落ちても書く）→ G1-SCR（落ちたら非 0・残りを起動しない）→ readout
            post = [post_job('scr_provenance', PY_SCR, ['src.act_chimera_scr_0913', '--provenance', '--outdir', str(out)]),
                    post_job('scr_g1', PY_SCR, ['src.act_chimera_scr_0913', '--g1', '--outdir', str(out)])]
            post += [post_job(f'scr_readout_{a}', PY_SCR, ['src.act_chimera_scr_0913', '--readout', a, '--outdir', str(out)])
                     for a in SCR_ARMS]
            run_queue(post, 'stage1-scr-post', rss, 4, env_name=env, rss_source=src)
        return st
    if env == 'rlmnist':
        dev = args.device
        if dev is None:
            raise SystemExit('--device cpu|cuda is required for RL (decided by the smoke timing; §2.3 未決 5)')
        rss, src = resolve_rss('rlmnist', args.rss_peak_gib, device=dev)     # 追補 2-3: その device の実測だけ
        cap = args.parallel
        if cap is None:
            thr = _load(SMOKE_LAUNCH / f'smoke-rlmnist_throughput_{dev}.json')
            if not thr or not thr.get('chosen_parallel'):
                raise StageRefused(f'RL parallel not measured on {dev}: run `smoke --env rlmnist --device {dev}` first or pass --parallel (§7)')
            cap = int(thr['chosen_parallel'])
        out = OUT / 'rlmnist'
        jobs = [rl_job(a, s, out, dev) for a in STAGE1_RL_ARMS for s in SEEDS]
        if not args.dry_run:
            _stage_prov('stage1-rlmnist', pre)
        st = run_queue(jobs, 'stage1-rlmnist', rss, min(cap, CAP['rlmnist']), dry_run=args.dry_run, env_name=env, rss_source=src)
        if not args.dry_run and queue_ok(st):
            post = [post_job(f'rl_merge_{a}', PY_B, ['src.act_chimera_rlmnist_0913', '--merge', '--arm', a, '--outdir', str(out)])
                    for a in STAGE1_RL_ARMS]
            if dev == 'cuda':                                 # GPU なら G1-RL を 3 seed × 50 タスク = 3000 セルに広げる（§2.3）
                post.append(post_job('rl_g1_full', PY_B, ['src.act_chimera_rlmnist_0913', '--g1-full', '--outdir', str(out)]))
            run_queue(post, 'stage1-rlmnist-post', rss, 2, env_name=env, rss_source=src)
        return st
    raise SystemExit(f'unknown env {env}')


def scr_lr0p005(args):
    pre = {} if args.dry_run else dict(committed=require_committed_code(), gates=require_gates('scr-lr0p005', 'scr'))
    rss, src = resolve_rss('scr', args.rss_peak_gib, horizon=10_000_000)   # 10M の外挿そのもの（×1.5 を重ねない）
    out = OUT / 'scr_lr0p005'
    jobs = [scr_job(a, out) for a in SCR_ARMS_LR0P005]
    if not args.dry_run:
        _stage_prov('scr-lr0p005', pre)
    st = run_queue(jobs, 'scr-lr0p005', rss, CAP['scr'], dry_run=args.dry_run, env_name='scr', rss_source=src)
    if not args.dry_run and queue_ok(st):
        post = [post_job('scr_provenance_lr0p005', PY_SCR, ['src.act_chimera_scr_0913', '--provenance', '--outdir', str(out)]),
                post_job('scr_g1_lr0p005', PY_SCR, ['src.act_chimera_scr_0913', '--g1', '--outdir', str(out)])]
        post += [post_job(f'scr_readout_{a}', PY_SCR, ['src.act_chimera_scr_0913', '--readout', a, '--outdir', str(out)])
                 for a in SCR_ARMS_LR0P005]
        run_queue(post, 'scr-lr0p005-post', rss, 4, env_name='scr', rss_source=src)
    return st


def stage2(args):
    env = args.env
    pre = {} if args.dry_run else dict(committed=require_committed_code(), gates=require_gates('stage2', env, args.device))
    if env == 'pmnist':
        rss, src = resolve_rss('pmnist', args.rss_peak_gib)
        jobs = []
        for lr in STAGE2_PM_LRS:
            out = OUT / 'stage2' / f'pmnist_lr{lr}'
            jobs += [pm_job(a, s, out, lr=lr) for a in ('LR', 'ELU1', 'SMAXH', 'SMINH') for s in SEEDS]
        if not args.dry_run:
            _stage_prov('stage2-pmnist', pre)
        return run_queue(jobs, 'stage2-pmnist', rss, CAP['pmnist'], dry_run=args.dry_run, env_name=env, rss_source=src)
    if env == 'rlmnist':
        if args.device is None:
            raise SystemExit('--device cpu|cuda is required')
        rss, src = resolve_rss('rlmnist', args.rss_peak_gib, device=args.device)   # 追補 2-3: その device の実測だけ
        out = OUT / 'rlmnist'
        thr = _load(SMOKE_LAUNCH / f'smoke-rlmnist_throughput_{args.device}.json') or {}
        cap = args.parallel or thr.get('chosen_parallel')
        if not cap:
            raise StageRefused('RL parallel not measured: run the RL smoke or pass --parallel (§7)')
        jobs = [rl_job(a, s, out, args.device) for a in STAGE2_RL_ARMS for s in SEEDS]
        if not args.dry_run:
            _stage_prov('stage2-rlmnist', pre)
        return run_queue(jobs, 'stage2-rlmnist', rss, min(int(cap), CAP['rlmnist']), dry_run=args.dry_run, env_name=env, rss_source=src)
    raise SystemExit(f'unknown env {env}')


def report(args):
    from src import act_chimera_report_0913 as RP
    argv = ['--root', str(OUT), '--pass', args.passes]
    if args.scr_lr0p005:
        argv.append('--scr-lr0p005')
    if args.stage2:
        argv.append('--stage2')
    RP.main(argv)


# ------------------------------------------------------------------ 片付け（CLAUDE.md §4・§8）
def backup(args):
    """git 外のファイル（__pycache__ 以外）を obsidian-research-data/<run_id>/ に移し、backup_manifest.json を書く。
    ``--execute`` が無ければ計画を印字するだけ。§8 で commit する出力（COMMIT_TARGETS）が未追跡のまま残っていれば
    ``--execute`` を拒否する（commit すべき rows・provenance・units を worktree の外へ動かさない）。"""
    import fnmatch
    dest = Path(args.dest).expanduser()
    r = subprocess.run(['git', 'status', '--porcelain', '--ignored', '--untracked-files=all'], cwd=ROOT, capture_output=True, text=True)
    files, commit_targets = [], []
    for line in r.stdout.splitlines():
        if line[:2].strip() not in ('??', '!!'):
            continue
        rel = line[3:].strip()
        p = ROOT / rel
        if '__pycache__' in rel or not p.is_file():
            continue
        if any(fnmatch.fnmatch(rel, pat) for pat in COMMIT_TARGETS):
            commit_targets.append(rel)
            continue
        if args.only and not rel.startswith(tuple(args.only.split(','))):
            continue
        files.append(rel)
    manifest = []
    for rel in files:
        src = ROOT / rel
        dst = dest / rel
        manifest.append(dict(source=str(src), backup=str(dst), bytes=src.stat().st_size, sha256=_sha(src)))
    if not args.execute:
        print(json.dumps(dict(n_files=len(files), total_bytes=sum(m['bytes'] for m in manifest), dest=str(dest), files=files[:50],
                              untracked_commit_targets=commit_targets[:50]), indent=1))
        return
    if commit_targets:
        raise StageRefused(f'backup --execute refused: {len(commit_targets)} files that §8 commits are untracked '
                           f'(git add them first): {commit_targets[:10]}')
    for m in manifest:
        dst = Path(m['backup'])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(m['source'], dst)
        assert _sha(dst) == m['sha256'], dst
    OUT.mkdir(parents=True, exist_ok=True)
    _dump(OUT / 'backup_manifest.json', dict(run_id=RUN_ID, time=time.strftime('%Y-%m-%dT%H:%M:%S'), git_head=git_head(), files=manifest))
    print(f'moved {len(manifest)} files -> {dest}; manifest {OUT / "backup_manifest.json"}')


def status(args):
    for d in (SMOKE_LAUNCH, LAUNCH):
        for p in sorted(d.glob('*_status.json')):
            st = json.loads(p.read_text())
            jobs = st.get('jobs', [])
            n = {k: sum(1 for j in jobs if j['status'] == k) for k in ('COMPLETE', 'FAILED', 'RUNNING', 'PENDING')}
            print(f"{p.relative_to(ROOT)}: {n} stopped={st.get('stopped_after_failure')} J_last={st['J_history'][-1] if st.get('J_history') else None}")
    print('active lock:', _load(ACTIVE_LOCK))
    print('MemAvailable GiB:', round(mem_available_gib(), 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('g0')
    p = sub.add_parser('smoke'); p.add_argument('--env', required=True, choices=['pmnist', 'scr', 'rlmnist'])
    p.add_argument('--device', default=None, choices=['cpu', 'cuda'])
    p.add_argument('--rss-peak-gib', type=float, default=None); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('g05'); p.add_argument('--root', default=None); p.add_argument('--scratch', default=None)
    for name in ('stage1', 'stage2'):
        p = sub.add_parser(name); p.add_argument('--env', required=True, choices=['pmnist', 'scr', 'rlmnist'])
        p.add_argument('--device', default=None, choices=['cpu', 'cuda']); p.add_argument('--parallel', type=int, default=None)
        p.add_argument('--rss-peak-gib', type=float, default=None); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('scr-lr0p005'); p.add_argument('--rss-peak-gib', type=float, default=None); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('report'); p.add_argument('--pass', dest='passes', default='both', choices=['1', '2', 'both'])
    p.add_argument('--scr-lr0p005', action='store_true'); p.add_argument('--stage2', action='store_true')
    p = sub.add_parser('backup'); p.add_argument('--dest', default=f'~/Projects/obsidian-research-data/{RUN_ID}')
    p.add_argument('--execute', action='store_true'); p.add_argument('--only', default='results/')
    sub.add_parser('status')
    args = ap.parse_args(argv)
    handlers = dict(g0=g0, smoke=smoke, g05=g05, stage1=stage1, stage2=stage2, report=report, backup=backup, status=status)
    handlers['scr-lr0p005'] = scr_lr0p005
    fn = handlers[args.cmd]
    if args.cmd in ('status',) or getattr(args, 'dry_run', False) or (args.cmd == 'backup' and not args.execute):
        return fn(args)
    with launcher_lock(' '.join(sys.argv[1:] if argv is None else argv)):       # サブコマンド全体で 1 つのロック
        return fn(args)


if __name__ == '__main__':
    main()

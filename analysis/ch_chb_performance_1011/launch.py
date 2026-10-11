"""Serial CH→CHB launcher; plan/check never query or initialize the GPU."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from analysis.ch_chb_performance_1011 import common as C

LOCK = Path("/tmp/lop_analysis_gpu.lock")
OWN_LOCK = Path(f"/tmp/{C.RUN}.lock")


def clean_committed():
    dirty = C.git("status", "--porcelain", "--untracked-files=all", "--", "src", "analysis", "specs")
    if dirty:
        raise ValueError(f"commit source and spec before running:\n{dirty}")
    head = C.git("rev-parse", "HEAD")
    if not C.git("branch", "-r", "--contains", head):
        raise ValueError("HEAD is not present in a remote branch; push the registration first")
    return head


def is_python_job(cmd, executable):
    """Only compute-context Python jobs block; ignore known desktop GPU clients."""
    if "--type=gpu-process" in cmd or "gnome-remote-desktop-daemon" in cmd:
        return False
    first = cmd.split("\0", 1)[0] if "\0" in cmd else cmd.split(" ", 1)[0]
    return any(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(v).name)
               for v in (first, executable))


def compute_jobs():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    jobs = []
    for line in result.stdout.splitlines():
        pid_s, _, name = line.partition(",")
        if not pid_s.strip().isdigit():
            continue
        pid = int(pid_s.strip())
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except FileNotFoundError:
            continue
        except PermissionError:
            cmd = name.strip()
        if pid != os.getpid() and is_python_job(cmd, name.strip()):
            jobs.append(dict(pid=pid, command=cmd.replace("\0", " ").strip()))
    return jobs


def await_gpu(lock, interval):
    if interval < 300:
        raise ValueError("GPU polling interval must be at least 300 seconds")
    while True:
        if (C.OUT / "STOP").exists():
            raise InterruptedError("STOP file present")
        acquired = False
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        jobs = compute_jobs() if acquired else None
        if acquired and not jobs:
            return
        if acquired:
            fcntl.flock(lock, fcntl.LOCK_UN)
        C.put(C.OUT / "launcher_state.json", dict(status="waiting_for_gpu", at=C.utc(),
                                                 pid=os.getpid(), compute_jobs=jobs,
                                                 shared_lock_busy=not acquired,
                                                 poll_interval_s=interval))
        print(f"{C.utc()} waiting for GPU: {jobs if jobs is not None else 'shared lock busy'}", flush=True)
        time.sleep(interval)


def verify_checks():
    checks = json.loads((C.OUT / "checks" / "cpu.json").read_text())
    if not checks.get("all_pass") or checks.get("source_sha256") != C.sources():
        raise ValueError("CPU checks missing, failed, or stale; run the check command")
    return C.sha(C.OUT / "checks" / "cpu.json")


def manifest_for(args, head):
    data = args.data_dir.resolve()
    return dict(config=C.config(), git_hash=head, prereg_commit=head,
                source_sha256=C.sources(), python=str(args.python.resolve()),
                data_dir=str(data), data_sha256={C.ARCHIVE: C.sha(data / C.ARCHIVE)},
                checks_sha256=verify_checks())


def prepare_manifest(args, head):
    wanted = manifest_for(args, head)
    path = C.OUT / "launch_manifest.json"
    if path.exists():
        if not args.resume:
            raise ValueError("existing launch requires explicit --resume")
        old = json.loads(path.read_text())
        for key, value in wanted.items():
            if old.get(key) != value:
                raise ValueError(f"resume manifest mismatch: {key}")
        return old
    if args.resume:
        raise ValueError("--resume requires this run's launch_manifest.json")
    allowed = {"checks", "logs"}
    unexpected = [p.name for p in C.OUT.iterdir() if p.name not in allowed] if C.OUT.exists() else []
    if unexpected:
        raise ValueError(f"refusing existing unregistered output: {unexpected}")
    wanted.update(created_at=C.utc(), command=sys.argv)
    C.put(path, wanted)
    return wanted


def validate_arm(arm, manifest):
    """Accept only a checkpoint created inside this registered new run."""
    import torch
    out = C.OUT / arm
    first = json.loads((out / "provenance_start.json").read_text())
    expected = dict(run_id=C.RUN, prereg_commit=manifest["prereg_commit"],
                    git_hash=manifest["git_hash"], arm=arm, seeds=C.SEEDS, conds=["raw"],
                    epochs=C.EPOCHS, tasks=C.TASKS, lr=C.LR, beta=C.BETA, lam=C.LAM[arm],
                    doors=C.DOORS[arm], threads=C.THREADS, device="cuda", dtype="float32",
                    data_sha256=manifest["data_sha256"])
    for key, value in expected.items():
        if first.get(key) != value:
            raise ValueError(f"{arm}: launch provenance mismatch: {key}")
    st = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    task = C.validate_checkpoint(st, arm, manifest["prereg_commit"])
    disk_rows = C.validate_rows(C.read_rows(out / "per_task.csv"), arm, task)
    checkpoint_rows = C.validate_rows(st["rows"], arm, task)
    for key, row in disk_rows.items():
        for metric in C.REQUIRED_METRICS:
            # The historical CSV writer uses ten significant digits.
            if abs(row[metric] - checkpoint_rows[key][metric]) > 6e-10 * max(1, abs(row[metric])):
                raise ValueError(f"{arm}: CSV/checkpoint disagreement: {key} {metric}")
    return task


def arm_worker(args):
    # Parent passes the held flock descriptor. Direct calls cannot bypass the GPU gate.
    if args.lock_fd is None or os.fstat(args.lock_fd).st_ino != LOCK.stat().st_ino:
        raise ValueError("internal worker requires the parent's GPU lock")
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((C.OUT / "launch_manifest.json").read_text())
    if C.sources() != manifest["source_sha256"] or clean_committed() != manifest["git_hash"]:
        raise ValueError("source changed since registration")
    import torch
    from src import relu_doors_0919 as E
    torch.set_num_threads(C.THREADS)
    E.RC.DATA_DIR = Path(manifest["data_dir"])
    cifar = E.RC.Cifar10()
    if cifar.sha256 != manifest["data_sha256"]:
        raise ValueError("CIFAR archive changed")
    out = C.OUT / args.arm
    resume = out.exists()
    if resume:
        done = validate_arm(args.arm, manifest)
        if done == C.TASKS:
            return finish_arm(args.arm, manifest)
    if (C.OUT / "STOP").exists():
        return 75
    def observer(event, state):
        if event == "task_end":
            C.put(out / "progress.json", dict(task=state["t"], at=C.utc(), pid=os.getpid()))
            return (C.OUT / "STOP").exists()
        return False
    E.run(args.arm, C.SEEDS, ["raw"], C.TASKS, C.EPOCHS, E.H.setup("cuda"), out,
          lr=C.LR, lam=C.LAM[args.arm], beta=C.BETA, checkpoint=True,
          resume=resume, cifar=cifar, graph=True, snapshots=True,
          run_id=C.RUN, prereg_commit=manifest["prereg_commit"], lifecycle=observer)
    return finish_arm(args.arm, manifest)


def finish_arm(arm, manifest):
    task = validate_arm(arm, manifest)
    out = C.OUT / arm
    status = "completed" if task == C.TASKS else "stopped"
    C.put(out / "completion.json", dict(status=status, completed_task=task,
                                         git_hash=C.git("rev-parse", "HEAD"), at=C.utc(),
                                         csv_sha256=C.sha(out / "per_task.csv"),
                                         checkpoint_sha256=C.sha(out / "ckpt.pt")))
    return 0 if task == C.TASKS else 75


def run(args):
    if args.poll_seconds < 300:
        raise ValueError("poll interval must be at least 300 seconds")
    with OWN_LOCK.open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare_manifest(args, clean_committed())
        (C.OUT / "logs").mkdir(parents=True, exist_ok=True)
        for arm in C.ARMS:
            if (C.OUT / arm).exists():
                validate_arm(arm, manifest)
        with LOCK.open("a") as lock:
            for arm in C.ARMS:
                if (C.OUT / "STOP").exists():
                    return 75
                if (C.OUT / arm).exists() and validate_arm(arm, manifest) == C.TASKS:
                    finish_arm(arm, manifest)
                    continue
                # Other experiments do not all cooperate with the shared flock.
                # Recheck and, if necessary, release/reacquire it before the second arm.
                fcntl.flock(lock, fcntl.LOCK_UN)
                await_gpu(lock, args.poll_seconds)
                C.put(C.OUT / "environment.json", dict(at=C.utc(),
                      gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                          "--format=csv"], check=True, capture_output=True, text=True).stdout,
                      CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")))
                command = [str(args.python), "-m", "analysis.ch_chb_performance_1011.launch",
                           "_arm", "--arm", arm, "--lock-fd", str(lock.fileno())]
                with (C.OUT / "logs" / f"{arm}.log").open("a", buffering=1) as log:
                    child = subprocess.Popen(command, cwd=C.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock.fileno(),))
                    C.put(C.OUT / "launcher_state.json", dict(status="running", arm=arm, pid=os.getpid(),
                                                              child_pid=child.pid, at=C.utc(), command=command))
                    rc = child.wait()
                if rc:
                    C.put(C.OUT / "launcher_state.json", dict(status="stopped" if rc == 75 else "failed",
                                                              arm=arm, exit_code=rc, at=C.utc()))
                    return rc
        from analysis.ch_chb_performance_1011.report import report
        report()
        C.put(C.OUT / "launcher_state.json", dict(status="completed", at=C.utc()))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "check", "run", "_arm"))
    parser.add_argument("--data-dir", type=Path, default=C.DATA)
    parser.add_argument("--python", type=Path, default=C.PYTHON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--arm", choices=C.ARMS)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps(dict(config=C.config(), output=str(C.OUT), data_dir=str(args.data_dir),
                              python=str(args.python), source_sha256=C.sources(),
                              execution="CH then CHB, shared GPU lock and external Python compute gate"), indent=2))
        return 0
    if args.mode == "check":
        from analysis.ch_chb_performance_1011.checks import run_checks
        return run_checks()
    if args.mode == "_arm":
        if not args.arm:
            parser.error("internal worker requires --arm")
        return arm_worker(args)
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InterruptedError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(75)

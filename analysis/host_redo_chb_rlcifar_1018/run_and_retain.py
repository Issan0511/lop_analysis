"""Host orchestration only: registered launcher, read-only verification, local backup.

This file is outside the reviewed scientific import closure. It does not import or
modify experiment code, acquire the GPU lock, alter an existing queue, or publish.
"""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = "redo_chb_rlcifar_1018"
OUT = ROOT / "results" / RUN
HOST = OUT / "host_continuation"
BACKUP = Path("/home/issan/Projects/obsidian-research-data") / RUN
PYTHON = Path("/home/issan/Projects/claude/proj_004_drift/.venv/bin/python")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def put(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def retain():
    # The training/report subprocesses have exited and all their logs are closed.
    # Freeze this list before writing the manifest, so it cannot include itself.
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    assert files and not any(p.is_symlink() for p in OUT.rglob("*")), "unexpected result symlink"
    needed = sum(p.stat().st_size for p in files)
    BACKUP.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(BACKUP).free > needed + (2 << 30), "insufficient backup space"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = BACKUP / ("host_gpu_" + stamp)
    destination.mkdir()  # Never overwrite a previous retained attempt.
    manifest = []
    for source in files:
        relative = source.relative_to(OUT)
        target = destination / "results" / RUN / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        size = source.stat().st_size
        shutil.copy2(source, target)
        assert target.stat().st_size == size and sha(target) == digest, str(relative)
        assert source.stat().st_size == size and sha(source) == digest, "source changed during backup"
        manifest.append(dict(source=str(source), backup=str(target), bytes=size, sha256=digest))
    record = dict(status="RETAINED_LOCALLY", verified_utc=now(), run_id=RUN,
                  source_root=str(OUT), destination=str(destination), files=len(manifest),
                  bytes=needed, manifest=manifest, originals_preserved=True,
                  github_published=False)
    put(destination / "backup_manifest.json", record)
    put(HOST / "backup_manifest.json", record)
    return dict(destination=str(destination), files=len(manifest), bytes=needed,
                manifest=str(HOST / "backup_manifest.json"))


def main():
    HOST.mkdir(parents=True, exist_ok=True)
    # A private controller lock prevents duplicate wrappers. The official CLI
    # alone owns /tmp/lop_analysis_gpu.lock for the whole sequential campaign.
    with (HOST / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = HOST / "state.json"
        plan = json.loads((HOST / "launch_plan.json").read_text())
        assert head() == plan["git_hash"], "HEAD differs from admitted launch plan"
        assert Path(sys.executable).resolve() == PYTHON.resolve(), "wrong Python"
        assert shutil.disk_usage(OUT).free >= (40 << 30), "insufficient campaign/backup space"
        state = dict(run_id=RUN, status="RUNNING", started_utc=now(), git_hash=head(),
                     service="redo-chb-rlcifar-1018.service", launch_plan=plan,
                     existing_queues_modified=False, github_published=False)
        put(state_path, state)
        try:
            command = [str(PYTHON), "-u", "-m", "analysis." + RUN + ".launch", "run", "--arm", "all"]
            with (HOST / "training.log").open("a") as log:
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            state.update(training_exit_code=code, training_finished_utc=now())
            if code == 0:
                assert head() == state["git_hash"], "HEAD changed during production"
                with (HOST / "verify.log").open("a") as log:
                    check = subprocess.run([str(PYTHON), "-m", "analysis." + RUN + ".report", "--verify-only"],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                state["verification_exit_code"] = check
                state["status"] = "COMPLETE" if check == 0 else "VERIFY_FAILED"
            else:
                state["status"] = "LAUNCHER_STOPPED_OR_FAILED"
            state["finished_utc"] = now()
        except Exception as exc:
            state.update(status="HOST_FAILED", error=repr(exc), finished_utc=now())
        put(state_path, state)
        try:
            retained = retain()
            put(HOST / "retention.json", dict(status="RETAINED_LOCALLY", verified_utc=now(), **retained))
        except Exception as exc:
            put(HOST / "retention.json", dict(status="RETENTION_FAILED", error=repr(exc), updated_utc=now()))
            raise
        print(json.dumps(dict(status=state["status"], retention=retained)), flush=True)
        return 0 if state["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

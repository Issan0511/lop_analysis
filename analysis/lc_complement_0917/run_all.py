"""Serial, durable execution of the already registered C1-C3 commands.

This driver changes no experimental parameters. Subprocess failures stop the
queue; incomplete shards are never skipped or treated as successful results.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLONE = Path.home() / "Projects/claude/proj_004_drift"
ARCHIVE = Path.home() / "Projects/obsidian-research-data/lc_complement_0917"
OUT = ARCHIVE / "main"
RESULTS = ROOT / "results/lc_complement_0917"
BRANCH = "codex/lc_complement_0917"


def now():
    return datetime.now(timezone.utc).isoformat()


def jobs():
    runner = [sys.executable, "-u", "-m", "src.lc_complement_0917"]
    verdict = [sys.executable, "-u", "-m", "analysis.lc_complement_0917.verdict"]
    for part, command in (("c1", "c1-ee"), ("c1_gpu", "c1-gpu"), ("c2", "c2"), ("c3", "c3")):
        if part == "c1_gpu":
            yield part, runner + [command, "--out", str(OUT / part)]
        elif part == "c3":
            for arm in ("E1", "S36", "C36", "E36", "E1_lr1e3"):
                for seed in range(3):
                    yield f"c3_{arm}_s{seed}", runner + [command, "--arm", arm, "--seed", str(seed),
                                                       "--out", str(OUT / part / arm / f"s{seed}")]
        else:
            for seed in range(10):
                yield f"{part}_s{seed}", runner + [command, "--seed", str(seed),
                                                  "--out", str(OUT / part / f"s{seed}")]
        yield f"{part}_aggregate", verdict + ["c1-gpu" if part == "c1_gpu" else part,
                                              "--input", str(OUT / part), "--out", str(RESULTS / part)]


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def save(state):
    state["updated_at"] = now()
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(OUT / "status.json")


def completed_prefix(state, plan):
    if state["status"] != "FAILED":
        raise ValueError("Explicit resume requires a failed, stopped run")
    names = [entry["job"] for entry in state["completed"]]
    if names != [name for name, _ in plan[:len(names)]] or len(names) >= len(plan):
        raise ValueError("Completed jobs must be an exact prefix of this plan")
    if state["current_job"] != plan[len(names)][0]:
        raise ValueError("Failure does not point to the next unfinished job")
    return len(names)


def resume(plan):
    from analysis.lc_complement_0917.verdict import load
    from src.lc_complement_0917 import code_hashes
    state = json.loads((OUT / "status.json").read_text())
    count = completed_prefix(state, plan)
    for name, command in plan[:count]:
        output = Path(command[command.index("--out") + 1])
        if name.endswith("_aggregate"):
            for file in ("summary.md", "verdict.csv", "verdict.json"):
                if not (output / file).is_file():
                    raise ValueError(f"Missing completed aggregate: {output / file}")
        else:
            p = json.loads((output / "provenance.json").read_text())
            if p["status"] != "COMPLETE" or p["smoke"] or p["code_sha256"] != code_hashes(p["part"]):
                raise ValueError(f"Completed shard is incompatible: {name}")
            if p["part"] != "c1-gpu":
                load(output, p["part"], p["config"]["seed"], p["config"].get("arm"))
    # Preserve failed data and logs verbatim; only the failed job is rerun.
    archive = OUT / "failed_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "status.json").write_text((OUT / "status.json").read_text())
    name, command = plan[count]
    failed_output = Path(command[command.index("--out") + 1])
    if failed_output.exists():
        failed_output.rename(archive / failed_output.name)
    failed_log = OUT / f"{name}.log"
    if failed_log.exists():
        failed_log.rename(archive / failed_log.name)
    state.setdefault("resumes", []).append(dict(time=now(), commit=git("rev-parse", "HEAD"),
                                                failed_attempt=str(archive)))
    state.update(status="RUNNING", pid=os.getpid())
    state.pop("error", None)
    return state, count


def finalize(state):
    # Small summaries go into git; all per-seed arrays and process logs already
    # live outside the worktree. Keep the original validation manifest entries.
    manifest_path = RESULTS / "backup_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {e["backup"]: e for e in manifest["files"]}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("status.json", "status.tmp", "supervisor.log", "launch.lock"):
            entries[str(p)] = dict(source=str(p), backup=str(p), bytes=p.stat().st_size,
                                   sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    manifest.update(scope="implementation validation and completed C1-C3 main runs", files=list(entries.values()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (RESULTS / "main_execution.json").write_text(json.dumps(state, indent=2) + "\n")
    summary = RESULTS / "summary.md"
    summary.write_text(summary.read_text() + "\n## 本走完了\n\n"
                       "C1 CPU10seed、C1 GPU150task、C2の10seed、C3の5腕×3seedが完了。"
                       "各部のsummary.mdとverdict.csvを参照。C4は既存の算術結果を使用。\n")
    git("add", "results/lc_complement_0917")
    git("commit", "-m", "Record completed LC complement C1-C3 runs and registered verdicts")
    for attempt in range(3):
        git("fetch", "origin")
        git("merge", "--no-edit", "origin/main")
        pushed = subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=ROOT)
        if pushed.returncode == 0:
            break
    else:
        raise RuntimeError("Could not push main after three fetch/merge attempts")
    state["result_commit"] = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", BRANCH, "origin/main")
    # Do not disturb somebody else's unsaved checkout or files.
    if not git("status", "--porcelain", cwd=CLONE):
        git("merge", "--ff-only", "origin/main", cwd=CLONE)
    for line in git("status", "--porcelain", "--ignored", "--untracked-files=all").splitlines():
        if not line.startswith("!! ") or ("__pycache__/" not in line and line[3:] != "data/mnist"):
            raise RuntimeError(f"Unarchived worktree file; leave worktree intact: {line}")
    link = ROOT / "data/mnist"
    if link.is_symlink() and link.resolve() == CLONE / "data/mnist":
        link.unlink()
    # Leave the current directory before asking git to remove this worktree.
    os.chdir(CLONE)
    git("worktree", "remove", str(ROOT), cwd=CLONE)
    git("branch", "-d", BRANCH, cwd=CLONE)
    git("push", "origin", "--delete", BRANCH, cwd=CLONE)


def main():
    if "--plan" in sys.argv:
        print(json.dumps(list(jobs()), indent=2))
        return
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = list(jobs())
    if "--resume" in sys.argv:
        state, offset = resume(plan)
    else:
        if (OUT / "status.json").exists():
            raise FileExistsError("Existing run status: refuse a duplicate or implicit restart")
        state = dict(status="RUNNING", started_at=now(), pid=os.getpid(),
                     run_commit=git("rev-parse", "HEAD"), completed=[], total_jobs=len(plan))
        offset = 0
    save(state)
    try:
        for name, command in plan[offset:]:
            state.update(current_job=name, current_started_at=now())
            save(state)
            log = OUT / f"{name}.log"
            print(f"{now()} START {name}", flush=True)
            with log.open("x") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            state["completed"].append(dict(job=name, finished_at=now(), log=str(log)))
            save(state)
            print(f"{now()} COMPLETE {name}", flush=True)
        state.update(status="RUNS_COMPLETE", current_job="finalize")
        save(state)
        finalize(state)
        state.update(status="COMPLETE", current_job=None, finished_at=now())
        save(state)
    except Exception as error:
        state.update(status="FAILED", error=repr(error))
        save(state)
        raise


if __name__ == "__main__":
    main()

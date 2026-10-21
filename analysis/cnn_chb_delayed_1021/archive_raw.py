#!/usr/bin/env python3
"""Dry-run by default. This helper is deliberately bound to ONE experiment.

python archive_raw.py                         # inventory only; no writes/locks
python archive_raw.py --execute              # verified copies + manifest
python archive_raw.py --execute --remove-sources  # additionally remove verified originals

Stage/commit intended result artifacts first. Execution requires both done
markers and COMPLETE_AUDITED report, no running producer locks, clean unstaged
tracked files, and no unexplained untracked/ignored path outside the two roots.
No git add/commit/clean/worktree/branch operation is performed.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile

RUN = "cnn_chb_delayed_1021"
WORKTREE = Path("/home/issan/Projects/claude/wt") / RUN
ARCHIVE = Path("/home/issan/Projects/obsidian-research-data") / RUN
COMMON_GIT = Path("/home/issan/Projects/claude/proj_004_drift/.git")
BRANCH = "codex/" + RUN
PREFIXES = (f"results/{RUN}/", f"analysis/{RUN}/")
MANIFEST_REL = f"results/{RUN}/backup_manifest.json"
ARMS = ("CH_continue", "CH_addB")


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def git(*args):
    return subprocess.check_output(["git", "-C", str(WORKTREE), *args])


def safe_relative(relative):
    path = PurePosixPath(relative)
    return (not path.is_absolute() and ".." not in path.parts and ".git" not in path.parts
            and str(path) == relative and any(relative.startswith(p) for p in PREFIXES))


def excluded_cache(relative):
    return "__pycache__" in PurePosixPath(relative).parts or relative.endswith(".pyc")


def assert_no_symlinks(path):
    """Check every existing component, including the fixed root's parents."""
    for component in (*reversed(path.parents), path):
        need(not component.is_symlink(), f"Symlink component refused: {component}")


def signature(path):
    assert_no_symlinks(path)
    value = path.lstat()
    need(stat.S_ISREG(value.st_mode), f"Non-regular file refused: {path}")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def digest(path):
    before = signature(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    need(signature(path) == before, f"File changed while hashing: {path}")
    return h.hexdigest(), before[2]


def validate_repository():
    assert_no_symlinks(WORKTREE)
    assert_no_symlinks(ARCHIVE)
    need(git("rev-parse", "--show-toplevel").decode().strip() == str(WORKTREE), "Wrong worktree")
    need(Path(git("rev-parse", "--git-common-dir").decode().strip()).resolve() == COMMON_GIT, "Wrong common git repository")
    need(git("branch", "--show-current").decode().strip() == BRANCH, "Wrong experiment branch")


def inventory():
    """ls-files expands ignored directories; status captures CLAUDE.md audit evidence."""
    tracked = set(git("ls-files", "-z").decode().split("\0")) - {""}
    status = git("status", "--porcelain=v1", "-z", "--ignored", "--untracked-files=all")
    paths = set()
    for options in (("--others", "--exclude-standard"), ("--others", "--ignored", "--exclude-standard")):
        paths.update(git("ls-files", *options, "-z").decode().split("\0"))
    rows, skipped, unexpected = [], [], []
    for rel in sorted(paths - {""}):
        if excluded_cache(rel):
            skipped.append(rel)
            continue
        if rel == MANIFEST_REL:
            continue  # The generated index is retained for git, never archived as raw data.
        if not safe_relative(rel):
            unexpected.append(rel)
            continue
        need(rel not in tracked, f"Tracked path in untracked inventory: {rel}")
        info = signature(WORKTREE / rel)
        rows.append(dict(source=str(WORKTREE / rel), source_relative=rel,
                         backup=str(ARCHIVE / rel), bytes=info[2],
                         lock_artifact=PurePosixPath(rel).name.endswith(".lock")))
    return dict(run_id=RUN, worktree=str(WORKTREE), archive=str(ARCHIVE),
                branch=BRANCH, head=git("rev-parse", "HEAD").decode().strip(),
                files=rows, file_count=len(rows), bytes=sum(r["bytes"] for r in rows),
                skipped_cache_count=len(skipped), unexpected_paths=unexpected,
                git_status_sha256=hashlib.sha256(status).hexdigest(),
                inventory_only=True, blocked=bool(unexpected))


def existing_entries():
    path = WORKTREE / MANIFEST_REL
    if not path.exists():
        return {}
    need(not path.is_symlink(), "Manifest symlink refused")
    value = json.loads(path.read_text())
    need(isinstance(value, dict) and value.get("run_id") == RUN, "Unexpected existing manifest format/run")
    result = {}
    for row in value["files"]:
        rel = row["source_relative"]
        need(safe_relative(rel) and rel != MANIFEST_REL, f"Unexpected manifest path: {rel}")
        need(row["source"] == str(WORKTREE / rel) and row["backup"] == str(ARCHIVE / rel), f"Manifest root mismatch: {rel}")
        need(rel not in result, f"Duplicate manifest entry: {rel}")
        need(digest(Path(row["backup"])) == (row["sha256"], row["bytes"]), f"Existing archive mismatch: {rel}")
        result[rel] = row
    return result


def read_saved(relative, prior):
    path = WORKTREE / relative
    if not path.exists():
        need(relative in prior, f"Required completion/audit evidence absent: {relative}")
        path = Path(prior[relative]["backup"])
    assert_no_symlinks(path)
    return json.loads(path.read_text())


def completion_guard(prior):
    result_root = f"results/{RUN}"
    report = read_saved(result_root + "/report/report.json", prior)
    need(report.get("experiment") == RUN and report.get("status") == "COMPLETE_AUDITED", "Independent report is not COMPLETE_AUDITED")
    need(report.get("primary_matches_registered") is True and all(report.get("audit", {}).values()) and bool(report.get("audit")), "Report audit checks missing/failed")
    need(read_saved(result_root + "/checks/preflight.json", prior).get("all_pass") is True, "GPU preflight not PASS")
    for arm in ARMS:
        done = read_saved(f"{result_root}/{arm}/done.json", prior)
        provenance = read_saved(f"{result_root}/{arm}/provenance.json", prior)
        identity = done.get("identity", {})
        need(provenance.get("status") == "COMPLETE" and provenance.get("identity") == identity, f"Incomplete/mismatched arm: {arm}")
        need(identity.get("arm") == arm and identity.get("steps") == 30000
             and identity.get("data", {}).get("seeds") == list(range(10, 20)), f"Invalid terminal identity: {arm}")


@contextlib.contextmanager
def producer_locks():
    """Lock existing producer artifacts without creating or writing their bytes."""
    handles = []
    try:
        for prefix in PREFIXES:
            root = WORKTREE / prefix
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.lock")):
                signature(path)
                stream = path.open("rb")
                handles.append(stream)
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f"Producer still owns lock: {path}") from error
        yield
    finally:
        for stream in reversed(handles):
            stream.close()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_verified(source, destination):
    before = signature(source)
    sha, size = digest(source)
    assert_no_symlinks(destination)
    if destination.exists():
        need(digest(destination) == (sha, size), f"Refusing to overwrite differing archive: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".archive-copy-", delete=False) as writer:
                temporary = Path(writer.name)
                with source.open("rb") as reader:
                    for block in iter(lambda: reader.read(2**20), b""):
                        writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            need(signature(source) == before and digest(temporary) == (sha, size), f"Source changed/copy mismatch: {source}")
            # Atomic no-overwrite publication. Existing files are never truncated.
            os.link(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    need(signature(source) == before and digest(destination) == (sha, size), f"Post-copy verification failed: {source}")
    return sha, size, before


def write_json_atomic(path, payload):
    assert_no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".archive-manifest-", delete=False) as writer:
            temporary = Path(writer.name)
            writer.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def execute(remove_sources):
    with producer_locks():
        prior = existing_entries()
        local_manifest = WORKTREE / MANIFEST_REL
        prior_meta = json.loads(local_manifest.read_text()) if local_manifest.exists() else None
        archive_manifest = ARCHIVE / "backup_manifest.json"
        if archive_manifest.exists():
            assert_no_symlinks(archive_manifest)
            archived_meta = json.loads(archive_manifest.read_text())
            need(prior_meta is not None and archived_meta == prior_meta,
                 "Archive manifest has no identical local counterpart; inspect before proceeding")
        completion_guard(prior)
        # Staged files are tracked and protected; unstaged tracked edits need review first.
        need(not git("diff", "--name-only").strip(), "Unstaged tracked modifications present")
        plan = inventory()
        need(not plan["unexpected_paths"], "Unexpected untracked/ignored paths: " + repr(plan["unexpected_paths"]))
        index_before = git("ls-files", "--stage", "-z")
        records, signatures = dict(prior), {}
        for row in plan["files"]:
            rel = row["source_relative"]
            sha, size, original = copy_verified(Path(row["source"]), Path(row["backup"]))
            entry = {key: row[key] for key in ("source", "source_relative", "backup")}
            entry.update(bytes=size, sha256=sha)
            need(rel not in prior or prior[rel] == entry, f"Previously archived source changed: {rel}")
            records[rel], signatures[rel] = entry, original
        current = inventory()
        need([r["source_relative"] for r in current["files"]] == [r["source_relative"] for r in plan["files"]]
             and not current["unexpected_paths"], "Inventory changed during copying")
        need(git("ls-files", "--stage", "-z") == index_before, "Git index changed during copying")
        payload = dict(run_id=RUN, worktree=str(WORKTREE), archive_root=str(ARCHIVE),
                       source_commit=prior_meta["source_commit"] if prior_meta else plan["head"],
                       files=[records[k] for k in sorted(records)],
                       verification="sha256+bytes verified before any source removal")
        manifest = WORKTREE / MANIFEST_REL
        manifest_tracked = MANIFEST_REL in set(git("ls-files", "-z").decode().split("\0"))
        if manifest_tracked:
            need(json.loads(manifest.read_text()) == payload, "Tracked manifest would change; refusing to edit tracked file")
        else:
            write_json_atomic(manifest, payload)
        write_json_atomic(ARCHIVE / "backup_manifest.json", payload)
        # Both durable manifest copies exist before the first source deletion.
        removed = 0
        if remove_sources:
            for rel in sorted(signatures, key=lambda s: (s.endswith(".lock"), s)):
                row = records[rel]
                source, backup = Path(row["source"]), Path(row["backup"])
                need(git("ls-files", "--stage", "-z") == index_before, "Git index changed before deletion")
                need(signature(source) == signatures[rel], f"Source changed before deletion: {rel}")
                need(digest(source) == digest(backup) == (row["sha256"], row["bytes"]), f"Final hash mismatch: {rel}")
                source.unlink()
                fsync_directory(source.parent)
                removed += 1
        return dict(status="ARCHIVED_VERIFIED", new_files=len(signatures), manifest_files=len(records),
                    removed_sources=removed, manifest=str(manifest), archive_manifest=str(ARCHIVE / "backup_manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--remove-sources", action="store_true")
    args = parser.parse_args()
    need(not args.remove_sources or args.execute, "--remove-sources requires --execute")
    validate_repository()
    result = execute(args.remove_sources) if args.execute else inventory()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("blocked"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

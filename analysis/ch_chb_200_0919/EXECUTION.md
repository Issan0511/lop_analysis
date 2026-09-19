# S3 execution handoff (2026-09-20 JST)

Issa approved the original preregistration and prediction candidate, then explicitly instructed `実行して`. Next instruction: `待機は１時間で`. Check GPU availability / running task progress hourly. No new permission is needed for implementation, checks, CH then CHB, reporting, commits, archive and CLAUDE.md §4 cleanup. Independent audit has not been performed.

Worktree: `/home/issan/Projects/claude/wt/ch_chb_200_0919`, branch `claude/ch_chb_200_0919`. Read `specs/spec_ch_chb_200_0919.md` and `/home/issan/Projects/claude/CLAUDE.md`. Do not alter registered criteria. This is the original t50 trajectory, not fresh seeds. Worktree creation date stays 0919.

Execution update 2026-09-20 02:00 JST: GPU became free. Both CH/CHB short GPU comparisons passed (continuous vs completed-checkpoint resume, diagnostic ON/OFF, and frozen original engine), all 75 mutations detected. Final check run adds explicit mutation identity coverage and t200-sized history save cost. Check `checks/checks.json` and launcher/heartbeat for current state. No main t51 outcomes were generated or read before these checks. Main launch follows passing final checks and commit. Original checkpoint/hist/snapshot copies verified against the parent manifest. `results/ch_chb_200_0919/registration.json` and `source_manifest.json` exist. `data` is a symlink to the sole clone's data directory; never traverse it when archiving.

Sources:
- `src/relu_doors_0919.py`: optional lifecycle events only; default training unchanged.
- `src/ch_chb_200_0919.py`: input preparation, state hashes, diagnostics, prefix enrichment, task observer.
- `analysis/ch_chb_200_0919/checks.py`: CPU and GPU tests, mutation evidence, real resume and frozen-engine comparisons.
- `analysis/ch_chb_200_0919/launch.py`: serial launcher, shared lock, STOP and startup provenance.
- `analysis/ch_chb_200_0919/report.py`: frozen endpoint arithmetic and report.

Next:
1. At the next hourly wake, inspect `nvidia-smi` / compute PIDs once. Desktop C+G apps are not research jobs; `launch.compute_pids()` filters known desktop contexts. If any research job remains, leave queued until next hour. No repeated polling.
2. With GPU free run `OPENBLAS_NUM_THREADS=2 python3 -u analysis/ch_chb_200_0919/checks.py > /tmp/ch_chb_200_0919_checks.log 2>&1` in this worktree. These short tests may be checked earlier when expected to finish. Read failures and fix implementation, never loosen preregistration. Tests use seeds 200–209 only. Check `checks/checks.json` all_pass and required mutation evidence; zero/vacuous groups are rejected. Check measured cost/resources.
3. Inspect and address all bugs. In particular verify arithmetic, exact old CSV cells, column definitions, independent C/pin expectation, startup provenance, fail-closed launch. Run appropriate checks again after changed code. Tests currently have not executed on GPU.
4. Commit tested implementation + compact evidence before main launch. `checks.json` binds all five source files by SHA256; source changes require rechecking. Launcher insists clean src/analysis. Keep raw .pt/.npz/snap/hist/log out of git. Commit registration/spec before launch, no edits to thresholds after observing results.
5. Launch `nohup env OPENBLAS_NUM_THREADS=2 python3 -u analysis/ch_chb_200_0919/launch.py > results/ch_chb_200_0919/logs/ch_chb_200_0919_launcher.log 2>&1 < /dev/null &` after creating logs directory. Validate actual PID, startup provenance and heartbeat. CH then CHB are serialized; other GPU research jobs prohibited. Do not read online accuracy or intermediate verdict. Only launcher.json / heartbeat task/status for hourly checks. STOP file at result root is respected on task boundaries.
6. Launcher runs report only after both arms reach t200. On completion inspect full registered verdict, all seed types, prediction scores, diagnostics, and issues. If launch/report fails fix safely with STOP and preserved checkpoints; record source changes / restarts. Do not silently claim success. Do not alter any parent result.
7. Commit compact results, move ignored/untracked raw (except pycache, excluding all symlinks) into `/home/issan/Projects/obsidian-research-data/ch_chb_200_0919/` preserving repository-relative paths. Verify each sha256/bytes and commit `backup_manifest.json`. Fetch/merge origin/main in this worktree, push HEAD:main, verify ancestry, then remove only this worktree and branch. Preserve all pushed history (no rebase/squash/amend). Remote branch deletion only if it exists. Final user response in Japanese: result, predictions, evidence limitations, main commit and paths. Pause the hourly heartbeat after completion.

Completion 2026-09-20: both arms t200; registered main INCONCLUSIVE (7 B_ROUTE_DELAY seed pairs, below required 9). CH 7 COLLAPSE / 3 UNCLASSIFIED; CHB 10 HOLDS. Final self-verification passed. No restarts or training-code changes after launch. Archive and main integration follow; do not rerun the experiment.

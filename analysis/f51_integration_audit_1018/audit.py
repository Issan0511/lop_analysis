#!/usr/bin/env python3
"""Independent G0 audit. CPU success never substitutes for CUDA reproduction."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


BASE = "063b7c6085340475549981494e713a7ceea6dbbb"
HISTORICAL = "0eff0529e9d22df7e6d8b6c29000c1611ca5306d"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_stage(repo: Path, output: Path, stage: str, device: str,
                  reference: Path | None = None) -> dict:
    target = output / (device + "_" + stage)
    target.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "src/cifar5p1_mlp_0920_checks.py", "--stage", stage,
               "--device", device, "--out", str(target)]
    if reference is not None:
        command = [sys.executable, str(Path(__file__).with_name("reference_stage.py")),
                   "--repo", str(repo), "--reference", str(reference),
                   "--stage", stage, "--device", device, "--out", str(target)]
    with (target / "stdout.log").open("w") as stream:
        result = subprocess.run(command, cwd=repo, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    evidence = target / "checks.json"
    content = json.loads(evidence.read_text()) if evidence.exists() else {}
    return dict(command=command, returncode=result.returncode,
                checks_json=str(evidence), checks_sha256=sha(evidence) if evidence.exists() else None,
                all_pass=result.returncode == 0 and content.get("all_pass") is True,
                details=content)


def historical_reference(repo: Path, historical: Path, output: Path) -> dict:
    """Re-execute the pre-intervention source, never substitute current-host rows.

    The 0920 cu126 artifacts differ in float64 effective-rank diagnostics under
    cu130 (already documented in spec_resp_cifar5p1_1007 §1.2/§7). Old-versus-new
    environment comparison reports these columns separately. All columns,
    including effective rank, remain byte-exact in S-nochange in one environment.
    """
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=historical, text=True).strip()
    assert head == HISTORICAL, "historical reference must use immutable pre-intervention commit"
    assert not subprocess.check_output(["git", "status", "--porcelain", "--", "src"],
                                       cwd=historical), "historical source tree is dirty"
    reference = output / "historical_current_runtime"
    reference.mkdir()
    evidence = {}
    for arm in ("R", "SNA"):
        target = reference / f"{arm}_std_lr0.0001"
        command = [sys.executable, "src/cifar5p1_mlp_0920.py", "run", "--arm", arm,
                   "--seeds", "0-9", "--cond", "std", "--tasks", "30", "--lr", "0.0001",
                   "--device", "cuda", "--threads", "2", "--out", str(target)]
        with (reference / f"{arm}.log").open("w") as stream:
            subprocess.run(command, cwd=historical, stdout=stream, stderr=subprocess.STDOUT, check=True)
        old = repo / "results/cifar5p1_mlp_0920" / target.name
        left = list(csv.DictReader((old / "per_task.csv").open()))
        right = list(csv.DictReader((target / "per_task.csv").open()))
        assert len(left) == len(right) == 300
        assert {(int(r['seed']), int(r['task'])) for r in right} == {(s,t) for s in range(10) for t in range(1,31)}
        assert all(x.keys() == y.keys() for x,y in zip(left,right))
        differences = [dict(seed=x['seed'], task=x['task'], column=k, historical=x[k], current=y[k])
                       for x,y in zip(left,right) for k in x if x[k] != y[k]]
        non_diagnostic = [x for x in differences if x['column'] not in ('eff_rank_l1','eff_rank_l2')]
        fresh_exact = (old / "fresh_control.csv").read_bytes() == (target / "fresh_control.csv").read_bytes()
        evidence[arm] = dict(command=command, all_pass=not non_diagnostic and fresh_exact,
                             non_diagnostic_differences=non_diagnostic, fresh_byte_exact=fresh_exact,
                             report_only_effective_rank_differences=differences,
                             outputs_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file()},
                             recorded_outputs_sha256={n:sha(old/n) for n in ('per_task.csv','fresh_control.csv','provenance.json')})
    result = dict(all_pass=all(x['all_pass'] for x in evidence.values()), reference=str(reference),
                  historical_commit=head, historical_source_sha256={str(p.relative_to(historical)):sha(p) for p in sorted((historical/'src').glob('*.py'))},
                  scope='Historical source re-execution under current runtime; exact all-column comparison in S-nochange. No tolerance enlarged.',
                  prior_environment_rule='specs/spec_resp_cifar5p1_1007.md §1.2 and §7', arms=evidence)
    (reference / 'reference_evidence.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-validation", action="store_true")
    parser.add_argument("--historical-repo", type=Path,
                        help="Optional clean detached checkout of 0eff0529 for same-runtime historical reproduction")
    args = parser.parse_args()
    repo = args.repo.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    sys.path.insert(0, str(repo))
    from src import cifar5p1_mlp_0920 as host

    inherited = set((repo / "src").glob("*.py"))
    inherited.update((repo / "analysis/cifar_ledger_0920").glob("*.py"))
    inherited.update((repo / "analysis/resp_cifar_ee_0920").glob("*.py"))
    hashes = {str(p.relative_to(repo)): sha(p) for p in sorted(inherited)}
    constants = dict(dims=list(host.DIMS), n_classes=host.N_CLASSES,
                     batch=host.BATCH, steps_per_task=host.STEPS_PER_TASK,
                     n_tasks=host.N_TASKS, lr=host.LR,
                     std_mean=list(host.STD_MEAN), std_std=list(host.STD_STD))
    expected = dict(dims=[3072, 100, 100, 100], n_classes=100, batch=32,
                    steps_per_task=780, n_tasks=30, lr=0.0001,
                    std_mean=[0.5071, 0.4865, 0.4409], std_std=[0.2673, 0.2564, 0.2762])
    base_in_main = subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "main"],
                                  cwd=repo, check=False).returncode == 0
    tree = ast.parse((repo / "src/cifar5p1_mlp_0920.py").read_text())
    syntax_valid = any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body)
    cpu = execute_stage(repo, args.out, "pilot", "cpu")
    cuda = torch.cuda.is_available()
    gpu_stages = {}
    reference_evidence = None
    if cuda and args.cuda_validation:
        if args.historical_repo is not None:
            reference_evidence = historical_reference(repo, args.historical_repo.resolve(), args.out)
        for stage in ("main", "iv", "doors"):
            reference = Path(reference_evidence['reference']) if reference_evidence and stage in ('iv', 'doors') else None
            gpu_stages[stage] = execute_stage(repo, args.out, stage, "cuda", reference)
            if not gpu_stages[stage]["all_pass"]:
                break
    full_cuda = set(gpu_stages) == {"main", "iv", "doors"} and all(
        x["all_pass"] for x in gpu_stages.values())
    blockers = []
    if not cuda:
        blockers.append("BLOCKED_CUDA: torch.cuda.is_available() is false; no NVIDIA device visible")
    elif not args.cuda_validation:
        blockers.append("CUDA_REPRODUCTION_NOT_RUN: pass --cuda-validation on the registered GPU runtime")
    elif not full_cuda:
        blockers.append("CUDA_HOST_CHECK_FAILED: inspect per-stage evidence")
    if not cpu["all_pass"]:
        blockers.append("CPU_HOST_CHECK_FAILED")
    if not base_in_main:
        blockers.append("AUDITED_BASE_NOT_IN_MAIN")
    if constants != expected or not syntax_valid:
        blockers.append("STATIC_CONTRACT_FAILED")
    if reference_evidence is not None and not reference_evidence['all_pass']:
        blockers.append("HISTORICAL_PRIMARY_REPRODUCTION_FAILED")
    all_pass = not blockers and full_cuda
    report = dict(schema_version=1, status="PASS" if all_pass else "BLOCKED_CUDA" if not cuda else "INCOMPLETE",
                  all_pass=all_pass,
                  independent_auditor="Codex /root (independent of six implementers)",
                  audited_commit=BASE, source_sha256=hashes,
                  main_ancestor_verified=base_in_main,
                  host_reproduction_pass=full_cuda,
                  mutation_checks_pass=cpu["all_pass"] and full_cuda,
                  cpu_checks_pass=cpu["all_pass"], constants=constants,
                  historical_current_runtime=reference_evidence,
                  checks=[dict(name="constants", pass_=constants == expected),
                          dict(name="local_main_ancestry", pass_=base_in_main),
                          dict(name="host_ast", pass_=syntax_valid),
                          dict(name="cpu_pilot", pass_=cpu["all_pass"], evidence=cpu),
                          dict(name="cuda_main_iv_doors", pass_=full_cuda, evidence=gpu_stages)],
                  environment=dict(python=sys.version, torch=torch.__version__,
                                   cuda_build=torch.version.cuda, cuda_available=cuda,
                                   nvidia_devices=[p for p in os.listdir("/dev") if "nvidia" in p]),
                  blockers=blockers, created_at=datetime.now(timezone.utc).isoformat(),
                  scope="G0 only. New six-run implementations and comparator reuse need their own independent checks.")
    destination = args.out / "integration_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], all_pass=all_pass,
                          cpu_checks_pass=report["cpu_checks_pass"], report=str(destination),
                          blockers=blockers), ensure_ascii=False))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

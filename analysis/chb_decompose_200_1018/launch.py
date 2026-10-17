"""Fail-closed production launcher: registered CUDA/R10/200 tasks, no CPU fallback."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch
from src import chb_decompose_200_1018 as D
from analysis.chb_decompose_200_1018 import common as C
from analysis.chb_decompose_200_1018.diagnostics import Observer, digest

LOCK = Path("/tmp/lop_analysis_gpu.lock")


def environment():
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        gpu = dict(returncode=smi.returncode, stdout=smi.stdout, stderr=smi.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = dict(error=str(exc))
    return dict(at=C.utc(), torch=torch.__version__, cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), python=sys.version,
                executable=sys.executable, platform=platform.platform(), nvidia_smi=gpu,
                nvidia_devices=[str(p) for p in Path("/dev").glob("nvidia*")],
                numpy=__import__("numpy").__version__)


def validate_checkpoint(st, arm, prereg):
    expected = dict(arm=arm, seeds=C.SEEDS, conds=["raw"], epochs=C.EPOCHS, lr=C.LR,
                    lam=0. if arm == "CH" else C.LAM, beta=C.BETA, doors=D.DOORS[arm],
                    perturb=None, nan_slot=None, run_id=C.RUN, prereg_commit=prereg)
    if st.get("meta") != expected:
        raise ValueError("CHECK_FAILED: checkpoint metadata")
    t = C.integer(st["t"])
    if not 1 <= t <= 200 or st["tc"] != t * 30000:
        raise ValueError("CHECK_FAILED: checkpoint clock")
    C.validate_rows(st["rows"], arm, t)
    if st["diverged"] or not st["alive"].all() or len(st["alive"]) != 10:
        raise ValueError("DIVERGED")
    shapes = [(10, 100, 3072), (10, 100), (10, 100, 100), (10, 100), (10, 10, 100), (10, 10)]
    for key in ("P", "m", "v"):
        if len(st[key]) != 6:
            raise ValueError("CHECK_FAILED: optimizer state missing")
        for p, shape in zip(st[key], shapes):
            if tuple(p.shape) != shape or p.dtype != torch.float32 or not torch.isfinite(p).all():
                raise ValueError("CHECK_FAILED: tensor shape/dtype/finiteness")
    for key in ("g_lab", "g_batch"):
        if set(st[key]) != set(C.SEEDS):
            raise ValueError("CHECK_FAILED: RNG seeds")
    if len(st["act_state"].get("m", [])) != 2 or st.get("extension_state") is None:
        raise ValueError("CHECK_FAILED: EMA/diagnostic state")
    for i in D.frozen_indices(arm):
        init = torch.stack([D.E.H.init_params(s, torch.device("cpu"), D.E.DIMS)[i].detach() for s in C.SEEDS])
        if not torch.equal(st["P"][i], init) or torch.count_nonzero(st["m"][i]) or torch.count_nonzero(st["v"][i]):
            raise ValueError("CHECK_FAILED: frozen bias/optimizer buffers")
    return t


def reference_static():
    """Static eligibility is necessary, never sufficient for reuse."""
    record = dict(status="INELIGIBLE", natural_reproduction=False, reasons=[], arms={})
    try:
        manifest = json.loads((C.REFERENCE / "launch_manifest.json").read_text())
        record["manifest_sha256"] = C.sha(C.REFERENCE / "launch_manifest.json")
        record["registration_commit"] = manifest["prereg_commit"]
        for path, expected in manifest["source_sha256"].items():
            if C.sha(C.ROOT / path) != expected:
                record["reasons"].append("source differs: " + path)
        for arm in ("CH", "CHB"):
            p = json.loads((C.REFERENCE / arm / "provenance.json").read_text())
            C.validate_rows(C.read_rows(C.REFERENCE / arm / "per_task.csv"), arm)
            expected = dict(arm=arm, conds=["raw"], seeds=C.SEEDS, R=10, lr=C.LR,
                            n_tasks=200, epochs_per_task=400, steps_per_task=30000,
                            dims=[3072,100,100,10], door_lam=0. if arm == "CH" else C.LAM,
                            door_beta=.01, optimizer="adam", weight_decay=0., device="cuda")
            for k, v in expected.items():
                if p.get(k) != v:
                    record["reasons"].append(f"{arm} configuration mismatch: {k}")
            if p["torch"] != torch.__version__:
                record["reasons"].append(f"{arm} numerical environment torch differs: {p['torch']} vs {torch.__version__}")
            record["arms"][arm] = dict(csv_sha256=C.sha(C.REFERENCE / arm / "per_task.csv"),
                provenance_sha256=C.sha(C.REFERENCE / arm / "provenance.json"), torch=p["torch"])
    except (OSError, ValueError, KeyError) as exc:
        record["reasons"].append(str(exc))
    record["static_eligible"] = not record["reasons"]
    record["reuse_qualified"] = False
    record["policy"] = "Do not mix unqualified historical means: rerun CH and CHB from task1 in this environment."
    return record


def prepare_launch(resume, data_dir, reference):
    reg = C.registration()
    if C.git("status", "--porcelain", "--", "src", "analysis", "specs"):
        raise ValueError("commit implementation/specs before production launch")
    checks = json.loads((C.OUT / "checks.json").read_text())
    if not checks.get("cpu_all_pass") or checks.get("source_sha256") != C.sources():
        raise ValueError("CPU checks missing, failed or stale")
    archive = data_dir / D.E.RC.ARCHIVE
    manifest = dict(config=C.config(), registration=reg, git_hash=C.git("rev-parse", "HEAD"),
                    source_sha256=C.sources(), checks_sha256=C.sha(C.OUT / "checks.json"),
                    data_dir=str(data_dir.resolve()), data_sha256={archive.name: C.sha(archive)},
                    reference_qualification=reference, adapted_run_sha256=D.ADAPTED_RUN_SHA256)
    path = C.OUT / "launch_manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if not resume:
            raise ValueError("existing launch requires --resume")
        for k, v in manifest.items():
            if old.get(k) != v:
                raise ValueError("resume manifest mismatch: " + k)
        return old
    if resume:
        raise ValueError("--resume requires existing launch manifest")
    C.put(path, manifest)
    return manifest


def production(args):
    # The actual attempted production command persists its exact registered box
    # and environment before refusing CUDA absence. No reduced experiment runs.
    reg = C.registration()
    checks = json.loads((C.OUT / "checks.json").read_text())
    if not checks.get("cpu_all_pass") or checks.get("source_sha256") != C.sources():
        raise ValueError("CPU checks missing, failed or stale before production attempt")
    env = environment()
    reference = reference_static()
    C.put(C.OUT / "reference_qualification.json", reference)
    status = dict(status="BLOCKED_CUDA" if not env["cuda_available"] else "PREFLIGHT",
                  attempted_at=C.utc(), command=sys.argv, config=C.config(), registration=reg,
                  cpu_checks_sha256=C.sha(C.OUT / "checks.json"), cpu_checks_pass=True,
                  source_sha256=C.sources(), environment=env,
                  reason="CUDA unavailable; registered CUDA/R10/200-task production was not started."
                         if not env["cuda_available"] else None)
    C.put(C.OUT / "execution_status.json", status)
    C.put(C.OUT / "provenance.json", status)
    if not env["cuda_available"]:
        C.put(C.OUT / "verdict.json", dict(status="BLOCKED_CUDA", label=None,
              reason=status["reason"], missing_arms=list(C.ARMS), hypothesis_evaluated=False))
        shutil.copy2(C.SPEC, C.OUT / "spec.md")
        (C.OUT / "summary.md").write_text("# chb_decompose_200_1018\n\nBLOCKED_CUDA. Production launch was attempted with the registered CUDA, R=10, seed10–19, 200-task box. No production task ran and no scientific verdict is available.\n\nCPU implementation and mutation checks are recorded separately. Historical CH/CHB references have not qualified for reuse; the launcher reruns both baselines before the three new arms.\n")
        print(json.dumps(status, indent=2)); return 78
    manifest = prepare_launch(args.resume, args.data_dir, reference)
    # Always rerun baseline arms: conservative, valid fallback when historical
    # sources/environment or natural reproduction are not qualified.
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             check=True, capture_output=True, text=True)
        if any(x.strip().isdigit() and int(x) != os.getpid() for x in smi.stdout.splitlines()):
            raise RuntimeError("GPU compute job active; scheduler must retry when idle")
        torch.set_num_threads(2)
        device = D.E.H.setup("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        settings = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                        float32_matmul_precision=torch.get_float32_matmul_precision(),
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        cudnn_benchmark=torch.backends.cudnn.benchmark,
                        CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        if settings != dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                            float32_matmul_precision="highest", deterministic_algorithms=True,
                            cudnn_benchmark=False, CUBLAS_WORKSPACE_CONFIG=":4096:8"):
            raise ValueError("numerical execution settings do not match registration")
        C.put(C.OUT / "numerical_environment.json", settings)
        D.E.RC.DATA_DIR = args.data_dir
        cifar = D.E.RC.Cifar10()
        if cifar.sha256 != manifest["data_sha256"]:
            raise ValueError("data hash mismatch")
        # GPU kernel/graph parity checks use calibration seeds and remain
        # separate from the scientific outcome. They must pass before task1.
        from analysis.chb_decompose_200_1018.checks import gpu_checks
        gpu_result = gpu_checks(device)
        C.put(C.OUT / "checks_cuda.json", gpu_result)
        if not gpu_result["all_pass"]:
            raise ValueError("CUDA S checks failed")
        for arm in C.ARMS:
            if (C.OUT / "STOP").exists():
                return 75
            out = C.OUT / arm
            cp = None
            if (out / "ckpt.pt").exists():
                cp = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
                done = validate_checkpoint(cp, arm, manifest["registration"]["commit"])
                if done == 200:
                    continue
            elif out.exists():
                raise ValueError("arm output without checkpoint; refusing overwrite")
            observer = Observer(out, C.SEEDS, cp)
            log = C.OUT / "logs" / f"{arm}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                D.run(arm, C.SEEDS, ["raw"], 200, 400, device, out, lr=C.LR,
                      lam=0. if arm == "CH" else C.LAM, beta=C.BETA,
                      checkpoint=True, resume=cp is not None, snapshots=True,
                      graph=True, cifar=cifar, run_id=C.RUN,
                      prereg_commit=manifest["registration"]["commit"], lifecycle=observer)
            cp = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
            if validate_checkpoint(cp, arm, manifest["registration"]["commit"]) != 200:
                return 75
            C.validate_rows(C.read_rows(out / "per_task.csv"), arm)
        from analysis.chb_decompose_200_1018.report import report
        report()
    status.update(status="COMPLETE", finished_at=C.utc())
    C.put(C.OUT / "execution_status.json", status)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "plan", "run", "report"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data-dir", type=Path, default=C.DATA)
    a = p.parse_args()
    if a.mode == "plan":
        print(json.dumps(dict(config=C.config(), registration=C.registration(), sources=C.sources()), indent=2)); return 0
    if a.mode == "check":
        from analysis.chb_decompose_200_1018.checks import run_checks
        return run_checks()
    if a.mode == "report":
        from analysis.chb_decompose_200_1018.report import report
        report(); return 0
    return production(a)


if __name__ == "__main__":
    sys.exit(main())

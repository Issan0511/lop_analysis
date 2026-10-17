#!/usr/bin/env python3
"""Registered F51 Normalize-and-Project; CPU is exclusively for synthetic S checks.

Production: python -m src.nap_cifar5p1_1018 run --phase calib --integration-audit PATH
All numerical constants, seeds, tasks and step counts are fixed by the preregistered spec.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

from src import cifar5p1_mlp_0920 as C
from src import pmnist_0905 as H
from src import rlcifar_mlp_battle_0918 as B

EXPERIMENT = "nap_cifar5p1_1018"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / EXPERIMENT
SPEC = ROOT / "specs" / f"spec_{EXPERIMENT}.md"
COMMON_SPEC = ROOT / "specs/spec_six_runs_common_1018.md"
DEFAULT_DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar100") / C.ARCHIVE
DEFAULT_AUDIT = Path("/tmp/issa_six_specs_1018/integration_audit.json")
EPS_LN, LR = 1e-5, 1e-4
CALIB_SEEDS, EVAL_SEEDS = list(range(100, 110)), list(range(10))
HOST_SOURCES = ("src/cifar5p1_mlp_0920.py", "src/pmnist_0905.py",
                "src/pmnist_rlcifar_0907.py", "src/rlcifar_mlp_battle_0918.py")
OWN_SOURCES = ("src/nap_cifar5p1_1018.py", "analysis/nap_cifar5p1_1018/checks.py",
               "analysis/nap_cifar5p1_1018/verdict.py")
SOURCES = HOST_SOURCES + ("src/pmnist_rlmnist_0906.py",) + OWN_SOURCES
CHECK_KEYS = ("S-host", "S-LN", "S-Wproject", "S-affine", "S-grad", "S-fresh", "S-stats", "S-admission")
CANDIDATE = dict(name="NaP", candidates=1, linear_bias=False, output_bias=False,
                 ln_affine=True, ln_eps=EPS_LN, ln_axis="per image, hidden=100",
                 projection="per-seed Frobenius sphere on W1/W2/W3 at t0",
                 affine_projection="joint gamma/beta sphere sqrt(100)", lr=LR,
                 dims=list(C.DIMS), cond="std", steps=780, tasks=30, batch=32, R=10)


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_hashes() -> dict:
    return {p: sha_file(ROOT / p) for p in SOURCES}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def finite_json(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def init_nap(seeds: list[int], device, dtype=torch.float32) -> list[torch.Tensor]:
    # Generate *all six* host tensors before discarding biases: later W streams match.
    original = [C.init_params("R", s, device) for s in seeds]
    W = [torch.stack([q[2*l] for q in original]).to(dtype=dtype) for l in range(3)]
    R, h = len(seeds), C.HIDDEN
    P = [W[0], torch.ones(R, h, device=device, dtype=dtype),
         torch.zeros(R, h, device=device, dtype=dtype), W[1],
         torch.ones(R, h, device=device, dtype=dtype),
         torch.zeros(R, h, device=device, dtype=dtype), W[2]]
    return [p.contiguous().requires_grad_(True) for p in P]


def nap_forward(P, X):
    """The only biases are the two LN offsets. Native LN and ReLU carry autograd."""
    W1, gamma1, beta1, W2, gamma2, beta2, W3 = P
    z1 = torch.bmm(X, W1.transpose(1, 2))
    u1 = F.layer_norm(z1, (z1.shape[-1],), eps=EPS_LN)
    n1 = u1 * gamma1[:, None, :] + beta1[:, None, :]
    a1 = F.relu(n1)
    z2 = torch.bmm(a1, W2.transpose(1, 2))
    u2 = F.layer_norm(z2, (z2.shape[-1],), eps=EPS_LN)
    n2 = u2 * gamma2[:, None, :] + beta2[:, None, :]
    a2 = F.relu(n2)
    return dict(z1=z1, n1=n1, a1=a1, z2=z2, n2=n2, a2=a2,
                logits=torch.bmm(a2, W3.transpose(1, 2)))


@torch.no_grad()
def project(P, radii, invalid, movement):
    """No cap/clamp: smaller and larger matrices both return to the t0 sphere.

    Undefined projection sets a permanent per-seed flag. A zero denominator is not
    repaired; CUDA-safe flags are inspected at the next task boundary and emitted.
    Adam moments and time are deliberately absent from this function.
    """
    for j, k in enumerate((0, 3, 6)):
        norm = torch.linalg.vector_norm(P[k], dim=(1, 2))
        invalid.logical_or_(norm == 0)
        scale = radii[j] / norm
        movement[:, j].add_((scale - 1).abs() * norm)
        P[k].mul_(scale[:, None, None])
    for j, (g, b) in enumerate(((1, 2), (4, 5))):
        norm = (P[g].square().sum(1) + P[b].square().sum(1)).sqrt()
        invalid.logical_or_(norm == 0)
        scale = math.sqrt(P[g].shape[1]) / norm
        movement[:, j+3].add_((scale - 1).abs() * norm)
        # One coefficient computed from both pre-projection tensors, applied once.
        P[g].mul_(scale[:, None])
        P[b].mul_(scale[:, None])


class State:
    """Actual host Adam equation, shared by NaP and the no-method R-fresh route."""
    def __init__(self, seeds, device, nap=True):
        self.nap, self.seeds, self.device = nap, seeds, device
        if nap:
            self.P = init_nap(seeds, device)
        else:
            original = [C.init_params("R", s, device) for s in seeds]
            self.P = [torch.stack([q[k] for q in original]).contiguous().requires_grad_(True)
                      for k in range(6)]
        self.initial = [p.detach().clone() for p in self.P]
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.radii = [torch.linalg.vector_norm(self.P[k].detach(), dim=(1, 2))
                      for k in (0, 3, 6)] if nap else []
        self.invalid = torch.zeros(len(seeds), dtype=torch.bool, device=device)
        self.movement = torch.zeros(len(seeds), 5, device=device)
        self.inv_c1 = torch.zeros((), device=device)
        self.inv_c2 = torch.zeros((), device=device)
        self.acc_sum = torch.zeros(len(seeds), device=device)
        self.bad = torch.zeros(len(seeds), dtype=torch.bool, device=device)
        self.tc = 0
        self.act = C.make_act("R")
        self.act.init_state(len(seeds), device, key=f"NaP-host|{seeds}")

    def forward(self, X):
        if self.nap:
            return nap_forward(self.P, X)
        z1, a1, z2, a2, logits = B.forward(self.P, X, self.act, train=True)
        return dict(z1=z1, n1=z1, a1=a1, z2=z2, n2=z2, a2=a2, logits=logits)

    def step(self, X, Y):
        out = self.forward(X)
        R, batch = Y.shape
        loss = F.cross_entropy(out["logits"].reshape(-1, C.N_CLASSES), Y.reshape(-1),
                               reduction="none").view(R, batch).mean(1)
        self.acc_sum.add_((out["logits"].detach().argmax(-1) == Y).float().mean(1))
        gradients = torch.autograd.grad(loss.sum(), self.P)
        with torch.no_grad():
            self.bad.logical_or_(~torch.isfinite(loss))
            for p, gr, mi, vi in zip(self.P, gradients, self.m, self.v):
                mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
                vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
                p.sub_(LR * (mi * self.inv_c1) / ((vi * self.inv_c2).sqrt() + 1e-8))
            if self.nap:
                project(self.P, self.radii, self.invalid, self.movement)

    @torch.no_grad()
    def reset(self):
        for dst, initial in zip(self.P, self.initial):
            dst.copy_(initial)
        for p in (*self.m, *self.v):
            p.zero_()
        self.invalid.zero_(); self.bad.zero_(); self.movement.zero_(); self.acc_sum.zero_()
        self.inv_c1.zero_(); self.inv_c2.zero_()
        self.tc = 0

    def mutable(self):
        return [*self.P, *self.m, *self.v, self.invalid, self.movement,
                self.acc_sum, self.bad, self.inv_c1, self.inv_c2]

    def checkpoint(self):
        return dict(P=[p.detach().cpu() for p in self.P],
                    initial=[p.cpu() for p in self.initial],
                    m=[p.cpu() for p in self.m], v=[p.cpu() for p in self.v],
                    radii=[p.cpu() for p in self.radii], tc=self.tc,
                    invalid=self.invalid.cpu(), bad=self.bad.cpu())


class Trainer:
    """One graph at original R=10 and batch=32; restore every mutable warmup tensor."""
    def __init__(self, state: State, X, Y, graph=True):
        self.state, self.X, self.Y = state, X, Y
        self.idx = torch.zeros(len(state.seeds), C.BATCH, dtype=torch.long, device=X.device)
        self.cg = None
        if graph and X.device.type == "cuda":
            saved = [p.detach().clone() for p in state.mutable()]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                state.inv_c1.fill_(1); state.inv_c2.fill_(1)
                for _ in range(3):
                    self.step()
            torch.cuda.current_stream().wait_stream(side)
            self.cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cg):
                self.step()
            with torch.no_grad():
                for p, old in zip(state.mutable(), saved):
                    p.copy_(old)

    def step(self):
        self.state.step(self.X[self.idx], self.Y[self.idx])

    def train(self, batches):
        s = self.state
        s.acc_sum.zero_(); s.bad.zero_(); s.movement.zero_()
        for j in range(batches.shape[1]):
            self.idx.copy_(batches[:, j])
            s.tc += 1
            s.inv_c1.fill_(1 / (1 - 0.9 ** s.tc))
            s.inv_c2.fill_(1 / (1 - 0.999 ** s.tc))
            self.cg.replay() if self.cg is not None else self.step()
        if self.X.device.type == "cuda":
            torch.cuda.synchronize()
        return s.acc_sum.detach().cpu() / batches.shape[1]


@torch.no_grad()
def evaluate(state, X, Y, task, seeds):
    z = state.forward(X)
    # Readouts cannot advance Adam, data streams, or model state.
    records, units = [], []
    acc = (z["logits"].argmax(-1) == Y).float().mean(1).cpu()
    for r, seed in enumerate(seeds):
        status = ("PROJECTION_UNDEFINED" if bool(state.invalid[r]) else
                  "DIVERGED" if bool(state.bad[r]) or
                  any(not bool(torch.isfinite(p[r]).all()) for p in state.P) else "COMPLETE")
        row = dict(seed=seed, slot=r, task=task, hard=int(task % 2 == 1), status=status,
                   memo=float(acc[r]) if status == "COMPLETE" else None)
        if status == "COMPLETE":
            for l in (1, 2):
                gate = (z[f"n{l}"][r] > 0).to(X.dtype)
                row[f"dead_frac_l{l}"] = float((gate.amax(0) < C.DEAD_TOL).float().mean())
                row[f"eff_rank_l{l}"] = H.eff_rank(z[f"a{l}"][r])
                for kind in ("z", "n"):
                    a = z[f"{kind}{l}"][r]
                    m, s = a.mean(0), a.std(0, correction=1)
                    valid = s > 0
                    rel = m[valid] / s[valid]
                    prefix = f"{'preLN' if kind == 'z' else 'postLN'}_l{l}"
                    row[prefix+"_m_median"] = float(m.median())
                    row[prefix+"_s_median"] = float(s.median())
                    row[prefix+"_r_median"] = float(rel.median()) if len(rel) else None
                    row[prefix+"_s_zero"] = int((~valid).sum())
                    row[prefix+"_gate_var_median"] = float(gate.var(0, correction=0).median())
                    for u in range(len(m)):
                        units.append(dict(seed=seed, task=task, layer=l, kind=kind, unit=u,
                                          m=float(m[u]), s=float(s[u]),
                                          r=float(m[u]/s[u]) if s[u] > 0 else None,
                                          zero_s_sign=int(torch.sign(m[u])) if s[u] == 0 else None,
                                          var_phi_prime=float(gate[:,u].var(correction=0))))
            for l, k in enumerate((0, 3, 6)):
                row[f"W{l+1}_fro"] = float(state.P[k][r].norm())
                row[f"W{l+1}_rho_t0"] = float(state.radii[l][r])
            for j, name in enumerate(("W1", "W2", "W3", "LN1", "LN2")):
                row[f"projection_path_{name}"] = float(state.movement[r,j])
        records.append(row)
    return finite_json(records), finite_json(units)


def check_audit(path: Path):
    if not path.is_file():
        return ["G0_MISSING"]
    try:
        audit = json.loads(path.read_text())
    except (ValueError, OSError):
        return ["G0_UNREADABLE"]
    errors = []
    for key in ("all_pass", "main_ancestor_verified", "host_reproduction_pass", "mutation_checks_pass"):
        if audit.get(key) is not True:
            errors.append("G0_"+key.upper())
    if not audit.get("independent_auditor") or not audit.get("audited_commit"):
        errors.append("G0_NO_INDEPENDENT_AUDITOR")
    if "/run5_nap" in audit.get("independent_auditor", ""):
        errors.append("G0_AUDITOR_IS_IMPLEMENTER")
    commit = audit.get("audited_commit", "")
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "main"],
                                    cwd=ROOT, capture_output=True).returncode:
        errors.append("G0_AUDITED_COMMIT_NOT_MAIN_ANCESTOR")
    if audit.get("schema_version") != 1:
        errors.append("G0_SCHEMA")
    for name in HOST_SOURCES:
        if audit.get("source_sha256", {}).get(name) != sha_file(ROOT/name):
            errors.append("G0_SOURCE_"+name)
    return errors


def validate_host_definition():
    required = dict(DIMS=(3072,100,100,100), N_CLASSES=100, BATCH=32,
                    STEPS_PER_TASK=780, N_TASKS=30, HARD_CLASSES=5,
                    PER_CLASS_TRAIN=500, PER_CLASS_TEST=100,
                    STD_MEAN=(.5071,.4865,.4409), STD_STD=(.2673,.2564,.2762))
    return ["HOST_CONSTANT_"+key for key,value in required.items() if getattr(C,key) != value]


def validate_checks(path):
    if not path.is_file():
        return ["CHECKS_MISSING"]
    ck = json.loads(path.read_text())
    errors = []
    if ck.get("all_pass") is not True or ck.get("production_ready") is not True:
        errors.append("CHECKS_NOT_PRODUCTION_READY")
    if ck.get("source_sha256") != source_hashes():
        errors.append("CHECK_SOURCE_MISMATCH")
    if any(not ck.get(k, {}).get("pass") for k in CHECK_KEYS):
        errors.append("CHECKS_REQUIRED_MISSING")
    return errors


@contextlib.contextmanager
def gpu_exclusive():
    with Path("/tmp/lop_analysis_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def preflight(phase, data, audit, checks, out):
    errors = []
    cuda = torch.cuda.is_available()
    if not cuda:
        errors.append("BLOCKED_CUDA")
    errors += check_audit(audit)
    errors += validate_checks(checks)
    errors += validate_host_definition()
    if not data.is_file():
        errors.append("DATA_MISSING")
    else:
        # Parsing/data verification is permitted without exposing intervention results.
        try:
            d = C.Cifar100(data)
            data_hash = d.sha256
        except Exception as e:
            errors.append("DATA_INVALID:"+str(e))
            data_hash = None
    if not data.is_file():
        data_hash = None
    selected = OUT/"calib/selected.json"
    if phase == "eval":
        if not selected.is_file():
            errors.append("CALIBRATION_NOT_FIXED")
        else:
            selection = json.loads(selected.read_text())
            if selection.get("candidate") != CANDIDATE or selection.get("source_sha256") != source_hashes():
                errors.append("CALIBRATION_MISMATCH")
            if selection.get("calibration_data_sha256") != data_hash:
                errors.append("CALIBRATION_DATA_MISMATCH")
            artifacts = selection.get("calibration_artifacts_sha256", {})
            if not artifacts or any(not (ROOT/name).is_file() or sha_file(ROOT/name) != sha
                                    for name,sha in artifacts.items()):
                errors.append("CALIBRATION_ARTIFACT_MISMATCH")
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(selected.relative_to(ROOT))],
                                     cwd=ROOT, capture_output=True).returncode == 0
            dirty = subprocess.run(["git", "diff", "HEAD", "--", str(selected.relative_to(ROOT))],
                                   cwd=ROOT, capture_output=True).stdout
            if not tracked or dirty:
                errors.append("CALIBRATION_NOT_COMMITTED")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "analysis", "specs"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        errors.append("UNCOMMITTED_IMPLEMENTATION")
    if (out/"per_task.csv").exists():
        errors.append("OUTPUT_EXISTS_REQUIRES_EXPLICIT_RESUME")
    p = dict(run_id=EXPERIMENT, phase=phase, status=errors[0] if errors else "READY",
             blockers=errors, cuda_available=cuda, torch=torch.__version__,
             cuda_build=torch.version.cuda, candidate=CANDIDATE, seeds=CALIB_SEEDS if phase=="calib" else EVAL_SEEDS,
             git_hash=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             spec_sha256=sha_file(SPEC), common_spec_sha256=sha_file(COMMON_SPEC),
             source_sha256=source_hashes(), integration_audit=str(audit),
             integration_audit_sha256=sha_file(audit) if audit.is_file() else None,
             checks_sha256=sha_file(checks) if checks.is_file() else None,
             data_sha256=data_hash, attempted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             prediction=dict(vs_R="IMPROVES_DIRECTION", vs_SNA="OPEN_COMPARISON", vs_KKT1="OPEN_COMPARISON"))
    write_json(out/"preflight.json", p)
    return p


def run_production(phase, data, out, admission):
    """Full 30 x 780 stream; no CLI permits smaller registered production runs."""
    if admission["blockers"]:
        raise RuntimeError("Cannot execute without all preflight proofs")
    device = H.setup("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    seeds = CALIB_SEEDS if phase == "calib" else EVAL_SEEDS
    cifar = C.Cifar100(data)
    X, Y = cifar.inputs("train", "std", device), cifar.train_y.to(device)
    Xte, Yte = cifar.inputs("test", "std", device), cifar.test_y.to(device)
    tr_rows, te_rows = C.class_rows(cifar.train_y), C.class_rows(cifar.test_y)
    plans = [C.task_plan(s) for s in seeds]
    streams = [H.stream("c51_batch", s) for s in seeds]
    model = State(seeds, device)
    trainer = Trainer(model, X, Y)
    out.mkdir(parents=True, exist_ok=True)
    (out/"spec.md").write_text(SPEC.read_text()+"\n"+COMMON_SPEC.read_text())
    torch.save(model.checkpoint(), out/"initial.pt")
    rows, units, batch_hashes = [], [], []
    saved29 = None
    t0 = time.time()
    for task in range(1, C.N_TASKS+1):
        tr = [torch.cat([tr_rows[c] for c in plans[r][task-1][1]]) for r in range(10)]
        batches = torch.stack([C.batch_indices(streams[r], tr[r]) for r in range(10)]).to(device)
        batch_hashes.append(dict(task=task, per_seed={str(s):sha_tensor(batches[r]) for r,s in enumerate(seeds)}))
        if task == 29:
            saved29 = batches.detach().clone()
        online = trainer.train(batches)
        diagnostics, per_unit = evaluate(model, torch.stack([X[q] for q in tr]),
                                         torch.stack([Y[q] for q in tr]), task, seeds)
        te = [torch.cat([te_rows[c] for c in plans[r][task-1][1]]) for r in range(10)]
        with torch.no_grad():
            test = (model.forward(torch.stack([Xte[q] for q in te]))["logits"].argmax(-1) ==
                    torch.stack([Yte[q] for q in te])).float().mean(1).cpu()
        for r, row in enumerate(diagnostics):
            row.update(arm="NaP", lr=LR, cond="std", online_acc=float(online[r]) if row["status"]=="COMPLETE" else None,
                       test_acc=float(test[r]) if row["status"]=="COMPLETE" else None)
        rows.extend(diagnostics); units.extend(per_unit)
        H.write_csv(out/"per_task.csv", rows)
        H.write_csv(out/"unit_diagnostics.csv", units)
        write_json(out/"batch_hashes.json", {"tasks":batch_hashes})
        ckpt = dict(state=model.checkpoint(), task=task,
                    stream_states=[s.get_state() for s in streams], saved29=saved29.cpu() if saved29 is not None else None,
                    admission=admission, per_task_sha256=sha_file(out/"per_task.csv"))
        torch.save(ckpt, out/"checkpoint.pt.tmp")
        (out/"checkpoint.pt.tmp").replace(out/"checkpoint.pt")
        print(f"NaP {phase} task {task}/30 online={float(online.mean()):.6f} elapsed={time.time()-t0:.1f}s", flush=True)
    # NaP's own fresh control resets all W, LN, Adam time/moments and t0 radii.
    continual = {r["seed"]:r["online_acc"] for r in rows if r["task"] == 29}
    model.reset()
    fresh_self = trainer.train(saved29)
    fs = []
    for r,s in enumerate(seeds):
        valid = not bool(model.invalid[r] | model.bad[r]) and all(bool(torch.isfinite(p[r]).all()) for p in model.P)
        fresh_status = "PROJECTION_UNDEFINED" if bool(model.invalid[r]) else "COMPLETE" if valid else "DIVERGED"
        fs.append(dict(seed=s, task=29, control="NaP_self", status=fresh_status,
                       fresh_online_acc=float(fresh_self[r]) if valid else None, continual_online_acc=continual[s],
                       gap_self=float(fresh_self[r])-continual[s] if valid and continual[s] is not None else None,
                       batch_sha256=sha_tensor(saved29[r])))
    H.write_csv(out/"fresh_self.csv", fs)
    # Shared R-fresh has the original six biased maps, no LN, no projection.
    del trainer, model
    host = State(seeds, device, nap=False)
    host_trainer = Trainer(host, X, Y)
    fresh_R = host_trainer.train(saved29)
    rf=[]
    for r,s in enumerate(seeds):
        valid = not bool(host.bad[r]) and all(bool(torch.isfinite(p[r]).all()) for p in host.P)
        rf.append(dict(seed=s, task=29, control="R_fresh_shared", status="COMPLETE" if valid else "DIVERGED",
                       fresh_online_acc=float(fresh_R[r]) if valid else None,
                       batch_sha256=sha_tensor(saved29[r])))
    H.write_csv(out/"fresh_control.csv", rf)
    status = "COMPLETE" if all(q["status"] == "COMPLETE" for q in rows+fs+rf) else "DIVERGED"
    provenance = dict(admission, status=status, runtime_s=time.time()-t0,
                      seed_slots=[dict(seed=s, slot=r) for r,s in enumerate(seeds)],
                      class_sha256={str(s):sha_tensor(torch.tensor(C.seed_classes(s), dtype=torch.int64)) for s in seeds},
                      initial_sha256=sha_file(out/"initial.pt"), checkpoint_sha256=sha_file(out/"checkpoint.pt"),
                      outputs_sha256={p.name:sha_file(p) for p in (out/"per_task.csv", out/"unit_diagnostics.csv",
                                                                   out/"batch_hashes.json", out/"fresh_self.csv", out/"fresh_control.csv")},
                      diagnostics=dict(s_correction=1, var_phi_prime_correction=0,
                                       m_s_r="unitwise before median", zero_s="null r and signed count",
                                       postLN_gate_variance="variance of pointwise ReLU derivative, not full LN Jacobian"),
                      tf32=False, deterministic=True, R=10, graph=True)
    write_json(out/"provenance.json", provenance)
    write_json(out/"verdict.json", dict(status=status, comparisons="OPEN_COMPARISON",
                                       note="Use verdict.py only after qualified R/SNA/KKT1 reuse"))
    (out/"summary.md").write_text(f"NaP {phase}: {status}. Full 30-task run, ten seeds.\n"
                                  "NaP-versus-SNA/KKT1 is an open comparison, not a paper-derived predicted rank.\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("preflight", "run", "select"))
    ap.add_argument("--phase", choices=("calib", "eval"), default="calib")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--integration-audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--checks", type=Path, default=OUT/"checks.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    out = a.out or OUT/a.phase/"NaP"
    if a.stage == "select":
        prov = OUT/"calib/NaP/provenance.json"
        if not prov.exists():
            raise SystemExit("CALIBRATION_INCOMPLETE")
        p = json.loads(prov.read_text())
        if p.get("status") != "COMPLETE" or p.get("seeds") != CALIB_SEEDS or p.get("source_sha256") != source_hashes():
            raise SystemExit("CALIBRATION_INVALID")
        from analysis.nap_cifar5p1_1018.verdict import seed_window, load_rows
        seed_window(load_rows(prov.parent/"per_task.csv"), CALIB_SEEDS)
        artifacts = {str(q.relative_to(ROOT)):sha_file(q) for q in prov.parent.iterdir() if q.is_file()}
        write_json(OUT/"calib/selected.json", dict(candidate=CANDIDATE, source_sha256=source_hashes(),
                                                  calibration_data_sha256=p["data_sha256"],
                                                  calibration_artifacts_sha256=artifacts,
                                                  rule="one preregistered candidate; no performance selection"))
        return
    admission = preflight(a.phase, a.data, a.integration_audit, a.checks, out)
    print(json.dumps({"status":admission["status"], "blockers":admission["blockers"]}))
    if admission["blockers"]:
        raise SystemExit(2)
    if a.stage == "run":
        with gpu_exclusive():
            run_production(a.phase, a.data, out, admission)


if __name__ == "__main__":
    main()

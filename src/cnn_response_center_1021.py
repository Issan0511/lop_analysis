#!/usr/bin/env python3
"""Matched-value c1 response intervention, extending saved RL-CIFAR LR/LRc states.

Real-data commands must only run after spec/code preregistration. The companion
checks module has a synthetic mode which neither reads CIFAR nor checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC
from src import rlcifar_cnn_0908 as CN
from src import sna_cnn_cause_1009 as E
from src import cnn_drive_verify_1009_lr as LRM
from src import cnn_center_1010 as CENTER

EXPERIMENT = "cnn_response_center_1021"
DEFAULT_LR = Path("/home/issan/Projects/obsidian-research-data/cnn_drive_verify_1009/results/cnn_drive_verify_1009/LR")
DEFAULT_LRC = Path("/home/issan/Projects/obsidian-research-data/cnn_center_1010/results/cnn_center_1010/LRc")
MAIN_ARMS = ("LR_self", "LR_swap", "LRc_self", "LRc_swap")
DIAG_STEPS = (0, 100, 750, 3000, 15000, 30000)
SLOPE = .1
SPEC_PATH = "specs/spec_cnn_response_center_1021.md"


def cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [cpu(v) for v in x]
    return x


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_hash(x):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            q = v.detach().cpu().contiguous()
            h.update(str((str(q.dtype), tuple(q.shape))).encode())
            h.update(q.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for a in v:
                visit(a)
        else:
            h.update(repr(v).encode())
    visit(x)
    return h.hexdigest()


def tensor_equal(a, b):
    return a.shape == b.shape and bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))))


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


@contextlib.contextmanager
def locked(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another run holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def phi(z):
    return torch.where(z > 0, z, SLOPE * z)


def dphi(z):
    return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, SLOPE))


class AnchorAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frozen, diff):
        return torch.where(diff == 0, frozen, frozen + diff)
    @staticmethod
    def backward(ctx, grad):
        return None, grad


def conv1(P, xb):
    R, B = xb.shape[:2]
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    return F.conv2d(x, P[0].reshape(R * 16, 3, 5, 5), P[1].reshape(-1), padding=2, groups=R)


class Field:
    """Frozen receiver/donor c1 maps; image-aligned inputs supplied by the engine."""
    def __init__(self, receiver_P, donor_P, mode="swap", gain=None):
        self.P0 = [p.detach().clone() for p in receiver_P[:2]]
        self.PD = [p.detach().clone() for p in donor_P[:2]]
        self.mode = mode
        self.gain = None if gain is None else gain.detach().clone()

    def apply(self, z, xb, xd):
        with torch.no_grad():
            z0, zd = conv1(self.P0, xb), conv1(self.PD, xd)
            a0 = phi(z0)
        if self.mode == "gain":
            gain = self.gain.reshape(1, -1, 1, 1)
            return AnchorAdd.apply(a0, gain * (phi(z) - a0)), z, gain * dphi(z)
        ze = zd + (z - z0)
        return AnchorAdd.apply(a0, phi(ze) - phi(zd)), ze, dphi(ze)


def forward(P, xb, act, field=None, xd=None):
    """Same grouped-convolution/baddbmm operations as E.forward, except c1 phi."""
    if field is None:
        out = E.forward(P, xb, act)
        return out, out[0], dphi(out[0])
    R, B = xb.shape[:2]
    z1 = conv1(P, xb)
    a1, ze, de = field.apply(z1, xb, xd)
    z2 = F.conv2d(F.max_pool2d(a1, 2, 2), P[2].reshape(R * 16, 16, 5, 5),
                  P[3].reshape(-1), padding=2, groups=R)
    a2 = act.phi(z2, 1)
    h = F.max_pool2d(a2, 2, 2).reshape(B, R, CN.FLAT).transpose(0, 1)
    z3 = torch.baddbmm(P[5][:, None, :], h, P[4].transpose(1, 2)); a3 = act.phi(z3, 2)
    z4 = torch.baddbmm(P[7][:, None, :], a3, P[6].transpose(1, 2)); a4 = act.phi(z4, 3)
    logits = torch.baddbmm(P[9][:, None, :], a4, P[8].transpose(1, 2))
    return (z1, a1, z2, a2, z3, a3, z4, a4, logits), ze, de


def read_source(root, arm, seeds):
    path = Path(root) / "state.pt"
    source_experiment = {"LR":"cnn_drive_verify_1009", "LRc":"cnn_center_1010"}[arm]
    source_rel = Path("results") / source_experiment / arm
    metadata_root = ROOT / source_rel
    manifest_path = ROOT / "results" / source_experiment / "backup_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {v["source"]:v for v in manifest["files"]}
    def verify_archived_file(actual, relative):
        entry=entries.get(str(relative))
        if entry is None:
            raise RuntimeError(f"No archive manifest entry for source: {relative}")
        digest=sha_file(actual)
        if digest!=entry["sha256"] or actual.stat().st_size!=entry["bytes"]:
            raise RuntimeError(f"Archived state/checkpoint differs from registered manifest: {actual}")
        return digest
    full_digest=verify_archived_file(path,source_rel/"state.pt")
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["t"] != 30 or st["meta"]["epochs"] != 400 or st["meta"]["lr"] != .001:
        raise RuntimeError(f"Unexpected source protocol: {path}")
    slots = [tuple(a) for a in st["meta"]["slots"]]
    inds = [slots.index((arm, s)) for s in seeds]
    full = st["state"]
    selected = {k: [q[inds].clone() for q in full[k]] for k in ("P", "m", "v")}
    selected.update(tc=full["tc"], bad=full["bad"][inds].clone(),
                    act={k: [q[inds].clone() for q in qs] for k, qs in full["act"].items()},
                    g_lab={s: full["g_lab"][s].clone() for s in seeds},
                    g_batch={s: full["g_batch"][s].clone() for s in seeds})
    ckpt_hashes = {}
    for j, s in enumerate(seeds):
        cp = Path(root) / "ckpt" / f"{arm}_seed{s}_t30.pt"
        cp_digest=verify_archived_file(cp,source_rel/"ckpt"/cp.name)
        one = torch.load(cp, map_location="cpu", weights_only=False)
        assert one["seed"] == s and one["arm"] == arm and one["tc"] == selected["tc"]
        for k in ("P", "m", "v"):
            assert all(tensor_equal(a[j], b) for a, b in zip(selected[k], one[k])), (cp, k)
        for k in one["act"]:
            assert all(tensor_equal(a[j], b) for a, b in zip(selected["act"][k], one["act"][k])), (cp, k)
        ckpt_hashes[str(cp)] = cp_digest
    assert selected["tc"] == 30 * 30000
    provenance_path=metadata_root/"provenance.json"
    table_path=metadata_root/"per_task.csv"
    archived_runtime=json.loads(provenance_path.read_text())
    if [tuple(x) for x in archived_runtime["slots"]]!=slots or archived_runtime["epochs"]!=400 or archived_runtime["n_tasks"]!=30 or archived_runtime["lr"]!=.001:
        raise RuntimeError(f"Registered source metadata disagrees with saved state: {provenance_path}")
    return selected, {"full_path": str(path), "full_sha256": full_digest, "checkpoints": ckpt_hashes,
                      "saved_provenance": st.get("gits", []), "source_slots": [list(s) for s in slots],
                      "archived_runtime": archived_runtime,
                      "provenance_path":str(provenance_path), "provenance_sha256":sha_file(provenance_path),
                      "reference_table_path":str(table_path), "reference_table_sha256":sha_file(table_path),
                      "archive_manifest_path":str(manifest_path), "archive_manifest_sha256":sha_file(manifest_path),
                      "archive_link_verified":True,
                      "selected_state_sha256": tree_hash(selected)}


class Data:
    def __init__(self, seeds, device, lr_root=DEFAULT_LR, lrc_root=DEFAULT_LRC, data_dir=None):
        self.seeds, self.device = list(seeds), device
        if data_dir is not None:
            RC.DATA_DIR = Path(data_dir)
        self.cifar = RC.Cifar10()
        self.center = CENTER.train_channel_mean(self.cifar).to(device)
        self.ids = torch.stack([CN.subset_idx(s) for s in seeds])
        assert all(len(torch.unique(row)) == CN.N_IMAGES for row in self.ids)
        self.Xraw = torch.stack([CN.images(self.cifar, row, device) for row in self.ids])
        self.Xcenter = self.Xraw - self.center.reshape(1, 1, 3, 1, 1)
        self.sources, self.source_meta = {}, {}
        for arm, root in (("LR", lr_root), ("LRc", lrc_root)):
            self.sources[arm], self.source_meta[arm] = read_source(root, arm, seeds)
            archived = self.source_meta[arm]["archived_runtime"]
            if archived.get("data_sha256") != self.cifar.sha256:
                raise RuntimeError(f"CIFAR archive does not match saved source provenance: {arm}")
        for k in ("g_lab", "g_batch"):
            for s in seeds:
                assert torch.equal(self.sources["LR"][k][s], self.sources["LRc"][k][s]), (k, s)
        self.identity = {"seeds": list(seeds), "data_sha256": self.cifar.sha256,
                         "global_image_ids_sha256": tree_hash(self.ids),
                         "center": self.center.cpu().tolist(), "sources": self.source_meta}

    def inputs(self, arm):
        return self.Xcenter if arm == "LRc" else self.Xraw


class Engine(LRM.BundleLR):
    def __init__(self, data, name, graph=True):
        receiver = "LRc" if name.startswith("LRc_") else "LR"
        donor = "LR" if receiver == "LRc" else "LRc"
        mode = name.split("_", 1)[1]
        if mode not in ("nat", "swap", "self", "gain"):
            raise ValueError(name)
        # Parent allocates the original deterministic streams and all Adam tensors.
        super().__init__([("LR", s) for s in data.seeds], data.cifar, data.device, graph=graph)
        self.name, self.data, self.receiver, self.donor = name, data, receiver, donor
        self.X = data.inputs(receiver)
        self.XD = data.inputs(receiver if mode == "self" else donor)
        self.load_full_state(data.sources[receiver])
        self.field = None
        if mode != "nat":
            dp = self.P if mode == "self" else [p.to(data.device) for p in data.sources[donor]["P"]]
            self.field = Field(self.P, dp, mode="gain" if mode == "gain" else "swap")
            if mode == "gain":
                self.field.gain = self.compute_gain()
        self.acc_steps = torch.zeros(self.R, CN.STEPS_PER_EPOCH, device=self.device)
        self.ce_steps = torch.zeros_like(self.acc_steps)

    @torch.no_grad()
    def compute_gain(self):
        """Channel gain matching the donor slope mean at initial receiver pool winners."""
        den = torch.zeros(self.R, 16, dtype=torch.float64, device=self.device)
        num = torch.zeros_like(den)
        for i in range(0, CN.N_IMAGES, CN.BATCH):
            xb, xd = self.X[:, i:i+CN.BATCH], self.XD[:, i:i+CN.BATCH]
            z0, zd = conv1(self.field.P0, xb), conv1(self.field.PD, xd)
            _, ix = F.max_pool2d(phi(z0), 2, 2, return_indices=True)
            for src, dst in ((z0, den), (zd, num)):
                d = dphi(src).flatten(2).gather(2, ix.flatten(2))
                dst += d.reshape(CN.BATCH, self.R, 16, -1).double().sum((0, 3))
        return (num / den).float()

    def forward_ids(self, ids):
        return forward(self.P, self.X[self.ar, ids], self.act, self.field,
                       self.XD[self.ar, ids] if self.field is not None else None)

    def step(self, j):
        idx = self.order[:, j * CN.BATCH:(j + 1) * CN.BATCH]
        out, _, _ = self.forward_ids(idx)
        logits, yb = out[8], self.Y[self.ar, idx]
        lossv = F.cross_entropy(logits.reshape(-1, 10), yb.reshape(-1), reduction="none").view(self.R, CN.BATCH).mean(1)
        hit = (logits.detach().argmax(-1) == yb).float().mean(1)
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            self.bad.logical_or_(~torch.isfinite(lossv))
            self.acc_steps[:, j].copy_(torch.where(self.bad, torch.zeros_like(hit), hit))
            self.ce_steps[:, j].copy_(torch.where(self.bad, torch.full_like(lossv, float("inf")), lossv))
            self.acc_ep.add_(self.acc_steps[:, j])
            self.tc.add_(1)
            c1 = (1 - torch.pow(self.b1, self.tc)).float(); c2 = (1 - torch.pow(self.b2, self.tc)).float()
            for p, gr, mi, vi in zip(self.P, grads, self.m, self.v):
                mi.mul_(.9).add_(gr, alpha=1-.9)
                vi.mul_(.999).addcmul_(gr, gr, value=1-.999)
                p.sub_(self.lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8))
            self.act.update([out[i].detach() for i in (0, 2, 4, 6)])

    def _state_tensors(self):
        return (*super()._state_tensors(), self.acc_steps, self.ce_steps)

    def snapshot(self):
        return cpu({"state": self.full_state(), "Y": self.Y, "order": self.order,
                    "acc_steps": self.acc_steps, "ce_steps": self.ce_steps,
                    "acc_ep": self.acc_ep,
                    "gain": self.field.gain if self.field is not None else None})

    def restore(self, st):
        self.load_full_state(st["state"])
        with torch.no_grad():
            for k in ("Y", "order", "acc_steps", "ce_steps", "acc_ep"):
                getattr(self, k).copy_(st[k])
        if st["gain"] is not None:
            self.field.gain = st["gain"].to(self.device)


@torch.no_grad()
def diagnose(B, update, chunk=CN.BATCH):
    """Full-image diagnostic; leaves all engine state and generators unchanged."""
    before = tree_hash(B.snapshot())
    R, dev = B.R, B.device
    # Per-channel/unit sums for raw sites and the actual artificial c1 coordinate.
    tags = list(CN.SITES) + ["c1_effective"]
    widths = list(CN.WIDTHS) + [16]
    sums = {tag: torch.zeros(R, w, 6, dtype=torch.float64, device=dev) for tag, w in zip(tags, widths)}
    counts = {tag: 0 for tag in tags}
    pools = {tag: torch.zeros(R, 16, 3, dtype=torch.float64, device=dev) for tag in ("c1", "c2", "c1_effective")}
    poolcounts = {tag: 0 for tag in pools}
    grams = [torch.zeros(R, w, w, dtype=torch.float64, device=dev) for w in CN.WIDTHS]
    correct = torch.zeros(R, dtype=torch.float64, device=dev); ce = torch.zeros_like(correct)
    for i0 in range(0, CN.N_IMAGES, chunk):
        n = min(chunk, CN.N_IMAGES - i0)
        ids = torch.arange(i0, i0+n, device=dev)[None, :].expand(R, -1)
        out, ze, de = B.forward_ids(ids)
        yb = B.Y[:, i0:i0+n]
        correct += (out[8].argmax(-1) == yb).sum(1)
        ce += F.cross_entropy(out[8].reshape(-1, 10), yb.reshape(-1), reduction="none").view(R, n).double().sum(1)
        for l, tag in enumerate(tags):
            z = ze if l == 4 else out[2*l]
            d = de if l == 4 else dphi(z)
            # Raw c1 derivative describes the original nonlinearity; c1_effective
            # is the derivative actually used by the artificial field.
            if l < 2 or l == 4:
                zz = z.reshape(n, R, 16, -1).permute(1, 2, 0, 3).reshape(R, 16, -1).double()
                dd = d.reshape(n, R, 16, -1).permute(1, 2, 0, 3).reshape(R, 16, -1).double()
                a = out[1] if l == 4 else out[2*l+1]
                _, ix = F.max_pool2d(a, 2, 2, return_indices=True)
                dp = d.flatten(2).gather(2, ix.flatten(2)).reshape(n, R, 16, -1).permute(1, 2, 0, 3).reshape(R, 16, -1).double()
                zp = z.flatten(2).gather(2, ix.flatten(2)).reshape(n, R, 16, -1).permute(1, 2, 0, 3).reshape(R, 16, -1)
                pools[tag] += torch.stack([dp.sum(-1), dp.square().sum(-1), (zp > 0).double().sum(-1)], -1)
                poolcounts[tag] += dp.shape[-1]
            else:
                zz, dd = z.transpose(1, 2).double(), d.transpose(1, 2).double()
            sums[tag] += torch.stack([zz.sum(-1), zz.square().sum(-1), (zz>0).double().sum(-1),
                                      dd.sum(-1), dd.square().sum(-1), (dd==0).double().sum(-1)], -1)
            counts[tag] += zz.shape[-1]
            if l < 4:
                a = out[2*l+1]
                ar = a.reshape(n, R, 16, -1).permute(1, 0, 3, 2).reshape(R, -1, 16).double() if l < 2 else a.double()
                grams[l] += ar.transpose(1, 2) @ ar
    rows, units = [], []
    metrics = {}
    for tag in tags:
        q = sums[tag] / counts[tag]
        sd = (q[...,1] - q[...,0].square()).clamp_min(0).sqrt()
        metrics[tag] = {"m": q[...,0], "sd": sd, "r": q[...,0]/sd.clamp_min(1e-30),
                        "p": q[...,2], "mean_d": q[...,3],
                        "var_d": (q[...,4]-q[...,3].square()).clamp_min(0), "zero_d": q[...,5]}
        if tag in pools:
            pp = pools[tag] / poolcounts[tag]
            metrics[tag].update(pool_mean_d=pp[...,0], pool_var_d=(pp[...,1]-pp[...,0].square()).clamp_min(0), pool_p=pp[...,2])
    for r, seed in enumerate(B.data.seeds):
        row = {"arm": B.name, "seed": seed, "update": update, "memo_acc": float(correct[r]/CN.N_IMAGES),
               "memo_ce": float(ce[r]/CN.N_IMAGES), "failed": bool(B.bad[r])}
        for l, tag in enumerate(tags):
            for k, val in metrics[tag].items():
                row[f"{tag}_{k}_unit_median"] = float(val[r].median())
                row[f"{tag}_{k}_unit_mean"] = float(val[r].mean())
            row[f"{tag}_p_below_005_fraction"] = float((metrics[tag]["p"][r] < .005).double().mean())
            if l < 4:
                G = grams[l][r].cpu()
                if torch.isfinite(G).all():
                    sv = torch.linalg.eigvalsh(G).clamp_min(0).sqrt()
                    prob = sv / sv.sum().clamp_min(1e-300)
                    erank = torch.exp(-(prob * prob.clamp_min(1e-300).log()).sum()) if sv.sum()>0 else torch.tensor(0.)
                    row[f"{tag}_effective_rank"] = float(erank)
                else:
                    row[f"{tag}_effective_rank"] = float("nan")
            for unit in range(widths[l]):
                units.append({"arm": B.name, "seed": seed, "update": update, "site": tag, "unit": unit,
                              **{k: float(val[r, unit]) for k, val in metrics[tag].items()}})
        rows.append(row)
    assert tree_hash(B.snapshot()) == before, "diagnostic changed training state"
    return rows, units


def numerical_context():
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "host": platform.node()}


def guard_release(release):
    """Refuse a draft/dirty execution and bind the released protocol exactly."""
    def git(*args):
        return subprocess.run(["git","-C",str(ROOT),*args],text=True,capture_output=True,check=True).stdout.strip()
    release_commit=git("rev-parse","--verify",release+"^{commit}")
    git("merge-base","--is-ancestor",release_commit,"HEAD")
    dirty=git("status","--porcelain","--untracked-files=all","--","src","specs")
    if dirty:
        raise RuntimeError("Uncommitted source/spec files invalidate preregistration:\n"+dirty)
    current=(ROOT/SPEC_PATH).read_text()
    registered=git("show",release_commit+":"+SPEC_PATH)
    if "登録状態: REGISTERED" not in current.splitlines() or "登録状態: REGISTERED" not in registered.splitlines():
        raise RuntimeError("The release spec must have 登録状態: REGISTERED; draft execution is forbidden")
    if "登録前の設計案" in current or "登録前の設計案" in registered:
        raise RuntimeError("Draft wording remains in the released protocol")
    if current.strip()!=registered.strip():
        raise RuntimeError("Current spec differs from --release; register its revision before running")
    return {"registered_spec_commit":release_commit,
            "implementation_commit":git("log","-1","--format=%H","--","src","specs"),
            "current_head":git("rev-parse","HEAD")}


def identity(data, steps, graph):
    paths = [Path(__file__), ROOT/"src/cnn_response_center_1021_checks.py",
             ROOT/"specs/spec_cnn_response_center_1021.md", ROOT/"src/sna_cnn_cause_1009.py",
             ROOT/"src/cnn_drive_verify_1009_lr.py", ROOT/"src/cnn_center_1010.py",
             ROOT/"src/rlcifar_cnn_0908.py", ROOT/"src/pmnist_rlcifar_0907.py", ROOT/"src/pmnist_0905.py"]
    return {"experiment": EXPERIMENT, "data": data.identity, "steps": steps, "graph": graph,
            "code_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in paths},
            "numerical_context": numerical_context(), "batch": CN.BATCH, "lr": .001}


def preflight_contract(data, graph, release_info):
    return {"contract_version":1,"registered_spec_commit":release_info["registered_spec_commit"],
            "implementation_commit":release_info["implementation_commit"],
            "identity":identity(data,30000,graph)}


def preflight_anchor(B, chunk=CN.BATCH):
    """All images: branch value equals natural receiver and pool winners match."""
    before = tree_hash(B.snapshot())
    maxdiff = [0.] * 9; winner_equal = True
    ce_diff = 0.
    with torch.no_grad():
        for i in range(0, CN.N_IMAGES, chunk):
            ids = torch.arange(i, min(i+chunk, CN.N_IMAGES), device=B.device)[None,:].expand(B.R, -1)
            out, _, _ = B.forward_ids(ids)
            ref = E.forward(B.P, B.X[B.ar, ids], B.act)
            for k, (a, b) in enumerate(zip(out, ref)):
                maxdiff[k] = max(maxdiff[k], float((a-b).abs().max()))
            for k in (1, 3):
                ia = F.max_pool2d(out[k], 2, 2, return_indices=True)[1]
                ib = F.max_pool2d(ref[k], 2, 2, return_indices=True)[1]
                winner_equal &= torch.equal(ia, ib)
            y = B.Y[B.ar, ids].reshape(-1)
            ca = F.cross_entropy(out[8].reshape(-1,10), y)
            cb = F.cross_entropy(ref[8].reshape(-1,10), y)
            ce_diff = max(ce_diff, float((ca-cb).abs()))
    assert tree_hash(B.snapshot()) == before
    return {"feature_max_abs": maxdiff, "ce_max_abs": ce_diff, "pool_winners_equal": winner_equal,
            "pass": max(maxdiff)==0. and ce_diff==0. and winner_equal}


def run_arm(data, arm, out, steps=30000, graph=True, stop_file=None):
    if steps % CN.STEPS_PER_EPOCH:
        raise ValueError("steps must end at an epoch boundary")
    armout = Path(out)/arm; armout.mkdir(parents=True, exist_ok=True)
    (armout/"checkpoints").mkdir(exist_ok=True)
    ident = identity(data, steps, graph)
    ident["arm"] = arm
    B = Engine(data, arm, graph=graph)
    statepath = armout/"state.pt"
    acc = np.zeros((B.R, steps), dtype=np.float32)
    ces = np.zeros_like(acc)
    epochrows, diagrows, unitrows = [], [], []
    start_epoch = 0
    if statepath.exists():
        saved = torch.load(statepath, map_location="cpu", weights_only=False)
        if saved["identity"] != ident:
            raise RuntimeError(f"Resume identity differs: {statepath}")
        B.restore(saved["engine"]); start_epoch = saved["epoch"]
        acc, ces = saved["acc"], saved["ce"]
        epochrows, diagrows, unitrows = saved["epochrows"], saved["diagrows"], saved["unitrows"]
    else:
        B.new_labels()
        pre = preflight_anchor(B)
        save_json(armout/"anchor.json", pre)
        if not pre["pass"]:
            raise RuntimeError(f"Anchor check failed: {arm}")
        dr, ur = diagnose(B, 0); diagrows += dr; unitrows += ur
        E.save_atomic(B.snapshot(), armout/"checkpoints"/"update00000.pt")
        E.save_atomic({"identity": ident, "engine": B.snapshot(), "epoch": 0,
                       "acc": acc, "ce": ces, "epochrows": epochrows,
                       "diagrows": diagrows, "unitrows": unitrows}, statepath)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    save_json(armout/"provenance.json", {"identity": ident, "git_hash": git,
              "command": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "resumed_at_epoch": start_epoch, "gain": cpu(B.field.gain).tolist() if B.field is not None and B.field.gain is not None else None})
    started = time.monotonic()
    for e in range(start_epoch, steps//CN.STEPS_PER_EPOCH):
        if stop_file is not None and Path(stop_file).exists():
            print(f"stop file found before epoch {e+1}; resume state retained", flush=True)
            break
        lo, hi = e*CN.STEPS_PER_EPOCH, (e+1)*CN.STEPS_PER_EPOCH
        interior = [k for k in DIAG_STEPS if lo < k < hi]
        t0 = time.monotonic()
        if interior:
            B.new_order(); B.acc_ep.zero_()
            for j in range(CN.STEPS_PER_EPOCH):
                B.step(j)
                k = lo+j+1
                if k in interior:
                    dr, ur = diagnose(B, k); diagrows += dr; unitrows += ur
                    E.save_atomic(B.snapshot(), armout/"checkpoints"/f"update{k:05d}.pt")
        else:
            B.run_epoch()
        acc[:, lo:hi] = B.acc_steps.detach().cpu().numpy()
        ces[:, lo:hi] = B.ce_steps.detach().cpu().numpy()
        for r, s in enumerate(data.seeds):
            epochrows.append({"arm": arm, "seed": s, "epoch": e+1, "update": hi,
                              "online_acc": float(acc[r,lo:hi].mean(dtype=np.float64)),
                              "online_ce": float(ces[r,lo:hi].mean(dtype=np.float64)), "failed": bool(B.bad[r])})
        if hi in DIAG_STEPS or hi == steps:
            dr, ur = diagnose(B, hi); diagrows += dr; unitrows += ur
            E.save_atomic(B.snapshot(), armout/"checkpoints"/f"update{hi:05d}.pt")
            write_csv(armout/"diagnostics.csv", diagrows); write_csv(armout/"unit_diagnostics.csv", unitrows)
        E.save_atomic({"identity": ident, "engine": B.snapshot(), "epoch": e+1,
                       "acc": acc, "ce": ces, "epochrows": epochrows,
                       "diagrows": diagrows, "unitrows": unitrows}, statepath)
        write_csv(armout/"epochs.csv", epochrows)
        print(f"{arm} epoch {e+1:03d}/{steps//75} update {hi:05d} online={acc[:,lo:hi].mean():.6f} wall={time.monotonic()-t0:.2f}s", flush=True)
    completed = len(epochrows)//B.R == steps//CN.STEPS_PER_EPOCH
    if completed:
        np.savez_compressed(armout/"batch_curves.npz", seeds=np.array(data.seeds), accuracy=acc, ce=ces)
        summaries = []
        for r, s in enumerate(data.seeds):
            summaries.append({"arm": arm, "seed": s, "first3000": float(acc[r,:min(3000,steps)].mean(dtype=np.float64)),
                              "all": float(acc[r].mean(dtype=np.float64)), "last3000": float(acc[r,-min(3000,steps):].mean(dtype=np.float64)),
                              "ce_first3000": float(ces[r,:min(3000,steps)].mean(dtype=np.float64)),
                              "ce_all": float(ces[r].mean(dtype=np.float64)), "ce_last3000": float(ces[r,-min(3000,steps):].mean(dtype=np.float64)),
                              "failed": bool(B.bad[r])})
        write_csv(armout/"summary.csv", summaries)
        save_json(armout/"done.json", {"identity": ident, "wall_this_process_s": time.monotonic()-started,
                                      "summary_sha256": sha_file(armout/"summary.csv")})
    del B
    if data.device.type == "cuda":
        torch.cuda.empty_cache()
    return completed


def sign_test(diffs):
    pos, neg = sum(bool(x > 0) for x in diffs), sum(bool(x < 0) for x in diffs)
    n = pos + neg
    p = min(1., 2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.
    return {"n_positive": pos, "n_negative": neg, "n_tie": len(diffs)-n, "p_two_sided": p}


def effect_direction(mean, positive, negative):
    mean_dir = "positive" if mean > 0 else "negative" if mean < 0 else "tie"
    sign_dir = "positive" if positive > negative else "negative" if positive < negative else "tie"
    direction = ("predicted" if mean_dir == sign_dir == "positive" else
                 "opposite" if mean_dir == sign_dir == "negative" else
                 "tie" if mean_dir == sign_dir == "tie" else "heterogeneous")
    return {"mean_direction": mean_dir, "sign_direction": sign_dir, "direction": direction}


def summarize(out):
    out = Path(out)
    tables = {}
    for arm in MAIN_ARMS:
        if not (out/arm/"done.json").exists():
            raise RuntimeError(f"Incomplete main arm: {arm}")
        with (out/arm/"summary.csv").open() as f:
            tables[arm] = {int(r["seed"]): r for r in csv.DictReader(f)}
    seeds = sorted(tables["LR_self"])
    assert all(sorted(v)==seeds for v in tables.values())
    if seeds != list(range(10,20)):
        raise RuntimeError("Main inference requires all 10 preregistered seeds")
    if any(json.loads((out/a/"done.json").read_text())["identity"]["steps"] != 30000 for a in MAIN_ARMS):
        raise RuntimeError("Main inference requires complete 30,000-update runs")
    pairs = (("rescue", "LR_swap", "LR_self"), ("reverse", "LRc_self", "LRc_swap"))
    rng = np.random.default_rng(1021)
    boots = rng.integers(0, len(seeds), size=(20000,len(seeds)))
    tests, rows = {}, []
    for label, a, b in pairs:
        diffs = np.array([float(tables[a][s]["first3000"])-float(tables[b][s]["first3000"]) for s in seeds])
        ci = np.percentile(diffs[boots].mean(1), [2.5,97.5]).tolist()
        tests[label] = {**sign_test(diffs), "mean_difference": float(diffs.mean()), "ci95_seed_bootstrap": ci,
                        "seed_differences": dict(zip(map(str,seeds), map(float,diffs)))}
        for s, d in zip(seeds,diffs):
            rows.append({"comparison": label, "seed": s, "plus_arm": a, "minus_arm": b,
                         "plus_first3000": tables[a][s]["first3000"], "minus_first3000": tables[b][s]["first3000"], "difference": float(d)})
    ordered = sorted(tests, key=lambda k: tests[k]["p_two_sided"])
    prev = 0.
    for i, k in enumerate(ordered):
        p = min(1., max(prev, (2-i)*tests[k]["p_two_sided"])); prev = p
        tests[k]["p_holm"] = p
        tests[k].update(effect_direction(tests[k]["mean_difference"], tests[k]["n_positive"], tests[k]["n_negative"]))
        tests[k]["significant"] = p <= .05
    supported = sum(t["significant"] and t["direction"]=="predicted" for t in tests.values())
    opposite = any(t["significant"] and t["direction"]=="opposite" for t in tests.values())
    heterogeneous = any(t["significant"] and t["direction"]=="heterogeneous" for t in tests.values())
    verdict = "HETEROGENEOUS" if heterogeneous else "OPPOSITE" if opposite else "BIDIRECTIONAL" if supported==2 else "ONE_DIRECTION" if supported==1 else "NONE"
    self_reference = {}
    for metric in ("first3000", "all", "last3000"):
        nd = np.array([float(tables["LRc_self"][s][metric])-float(tables["LR_self"][s][metric]) for s in seeds])
        self_reference[metric] = {"mean_LRc_self_minus_LR_self": float(nd.mean()),
                                  "ci95_seed_bootstrap": np.percentile(nd[boots].mean(1),[2.5,97.5]).tolist(),
                                  "seed_differences": dict(zip(map(str,seeds),map(float,nd)))}
    save_json(out/"verdict.json", {"primary": verdict, "tests": tests, "alpha": .05,
                                   "family": "two primary directions", "seeds": seeds,
                                   "self_control_gap_secondary_reference": self_reference})
    write_csv(out/"paired_primary.csv", rows)
    write_csv(out/"all_seed_summary.csv", [tables[a][s] for a in MAIN_ARMS for s in seeds])
    print(json.dumps({"primary":verdict,"tests":tests},ensure_ascii=False,indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run","summarize"))
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="10-19")
    p.add_argument("--arms", default=",".join(MAIN_ARMS))
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr-root", default=str(DEFAULT_LR)); p.add_argument("--lrc-root", default=str(DEFAULT_LRC))
    p.add_argument("--data-dir")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--stop-file")
    p.add_argument("--gpu-lock", default="/tmp/lop_analysis_gpu.lock")
    p.add_argument("--checks", help="Passed real preflight JSON (defaults to OUT/checks/checks.json)")
    p.add_argument("--release", required=False, help="Preregistered spec/code commit; required for real data")
    return p


def main():
    args = parser().parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "summarize":
        summarize(out); return
    if not args.release:
        raise SystemExit("Real-data execution requires --release <preregistered commit>")
    release_info=guard_release(args.release)
    if args.steps!=30000 or E.parse_seeds(args.seeds)!=list(range(10,20)):
        raise SystemExit("Registered run fixes steps=30000 and seeds=10-19 in one bundle")
    if not set(args.arms.split(",")) <= set(MAIN_ARMS):
        raise SystemExit("Only the four preregistered main arms can run")
    device = H.setup(args.device)
    with locked(out/".run.lock"), locked(args.gpu_lock):
        data = Data(E.parse_seeds(args.seeds), device, args.lr_root, args.lrc_root, args.data_dir)
        checkpath = Path(args.checks) if args.checks else out/"checks"/"checks.json"
        checked = json.loads(checkpath.read_text())
        if not checked.get("all_pass") or checked["source_states_and_streams"]["identity"] != data.identity:
            raise RuntimeError(f"Missing matching successful real preflight: {checkpath}")
        if checked.get("preflight_contract",{}).get("value") != preflight_contract(data,not args.eager,release_info):
            raise RuntimeError("Preflight code/spec/environment/bundle differs; run checks again after registration")
        for arm in args.arms.split(","):
            current_release=guard_release(args.release)
            if checked["preflight_contract"]["value"] != preflight_contract(data,not args.eager,current_release):
                raise RuntimeError("Code/spec/environment changed before an arm; stop and repeat preflight")
            if not run_arm(data, arm, out, args.steps, not args.eager, args.stop_file):
                break


if __name__ == "__main__":
    main()

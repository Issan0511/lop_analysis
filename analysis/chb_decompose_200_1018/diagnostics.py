"""Read-only task and exact-step diagnostics, with graph warmup reset."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import torch
from src import chb_decompose_200_1018 as D
from analysis.chb_decompose_200_1018 import common as C


def digest(obj):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                add(k); add(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x: add(y)
        else:
            h.update((type(x).__name__ + repr(x)).encode())
    add(obj)
    return h.hexdigest()


def gamma(n, u):
    if n * u >= 1:
        raise ValueError("invalid floating-point error budget")
    return n * u / (1 - n * u)


@torch.no_grad()
def units(P, X, act):
    z1, _, z2, _, _ = D.E.forward(P, X, act, train=False)
    out = {}
    for li, z in enumerate((z1, z2)):
        # Preserve the host clamp's native autograd convention at exactly zero
        # (gradient 1). Its legacy dphi diagnostic uses the strict z>0 gate.
        with torch.enable_grad():
            native_z = z.detach().requires_grad_(True)
            value = act.phi(native_z, li, train=False)
            native_d = torch.autograd.grad(value, native_z, torch.ones_like(value))[0].detach().double()
        z = z.double()
        m, sd = z.mean(1), z.std(1, correction=1)
        r = m / sd  # Preserve +/-inf and 0/0 in NPZ, with explicit counts below.
        dr = (z > 0).double()
        values = dict(b=P[2*li+1].double(), m=m, s=sd, r=r,
                      derivative_mean=native_d.mean(1), derivative_var=native_d.var(1, correction=0),
                      legacy_gate_mean=dr.mean(1), legacy_gate_var=dr.var(1, correction=0),
                      dead=(dr.amax(1) == 0), w_norm=P[2*li].double().norm(dim=2))
        out.update({f"l{li+1}_{k}": v.cpu().numpy() for k, v in values.items()})
    # Independent double-precision reference and a priori error bound for the
    # float32 affine dot products (no tolerance chosen from observed residual).
    xx, ww, bb = X.double(), P[0].double(), P[1].double()
    expected = torch.bmm(ww, xx.mean(1)[:, :, None]).squeeze(2) + bb
    reference_abs = torch.bmm(ww.abs(), xx.abs().mean(1)[:, :, None]).squeeze(2) + bb.abs()
    bound = gamma(X.shape[2]+2, 2**-24) * reference_abs
    bound += gamma(X.shape[1]+X.shape[2]+4, 2**-53) * (reference_abs + z1.double().abs().mean(1))
    residual = (z1.double().mean(1) - expected).abs()
    if not torch.all(residual <= bound):
        raise ValueError("CHECK_FAILED: first-layer mean identity")
    out.update(input_center_residual=xx.mean(1).cpu().numpy(),
               l1_mean_expected=expected.cpu().numpy(), l1_identity_residual=residual.cpu().numpy(),
               l1_identity_bound=bound.cpu().numpy())
    return out


class Observer:
    """The only accumulated state is diagnostic; host warmup is reset at ready."""
    def __init__(self, out, seeds, expected_checkpoint=None):
        self.out, self.seeds = Path(out), list(seeds)
        self.expected_checkpoint = expected_checkpoint
        self.accum = self.n = None
        self.rows = C.read_rows(self.out / "diagnostics.csv") if (self.out / "diagnostics.csv").exists() else []
        self.streams = []
        if (self.out / "streams.json").exists():
            import json
            self.streams = json.loads((self.out / "streams.json").read_text())

    def checkpoint_state(self):
        return dict(accum=self.accum.detach().clone(), count=self.n.detach().clone(),
                    previous_rng=self.prev_rng)

    def parameter_step(self, pi, p, mi, vi, inv_c1, inv_c2, lr, arm):
        if pi not in (0, 2, 4):
            return
        if self.accum is None:
            self.accum = torch.zeros(p.shape[0], 3, 5, dtype=torch.float64, device=p.device)
            self.n = torch.zeros((), dtype=torch.int64, device=p.device)
        with torch.no_grad():
            candidate = p - lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + 1e-8)
            final = candidate * (1 - lr * D.LAM) if pi in D.wd_indices(arm) else candidate
            adam_norm = (p-candidate).flatten(1).norm(dim=1)
            wd_norm = (candidate-final).flatten(1).norm(dim=1)
            actual_norm = (p-final).flatten(1).norm(dim=1)
            old_norm = p.flatten(1).norm(dim=1)
            relative = torch.where(old_norm > 0, actual_norm / old_norm, torch.zeros_like(old_norm))
            values = (adam_norm, wd_norm, actual_norm, relative, (old_norm == 0).to(p.dtype))
            for k, v in enumerate(values):
                self.accum[:, pi//2, k].add_(v.double())
            if pi == 0:
                self.n.add_(1)

    def __call__(self, event, c):
        if event == "ready":
            self.Xhash = digest(c["X"])
            self.arm = c["arm"]
            self.freeze = {i: c["P"][i].detach().clone() for i in D.frozen_indices(self.arm)}
            self.prev_rng = {k: {s: g.get_state() for s, g in c[k].items()} for k in ("g_lab", "g_batch")}
            actual = dict(P=c["P"], m=c["adam_m"], v=c["adam_v"], tc=c["tc"],
                          act_state=c["act"].state(), alive=c["alive"], **self.prev_rng)
            if self.expected_checkpoint:
                old = {k: self.expected_checkpoint[k] for k in actual}
                if digest(actual) != digest(old):
                    raise ValueError("CHECK_FAILED: checkpoint continuity before first draw")
            self.ready_digest = digest(actual)
            C.put(self.out / "continuity.json", dict(before_first_draw_sha256=self.ready_digest,
                  source_checkpoint_sha256=digest(old) if self.expected_checkpoint else None,
                  task=c["t_first"], initial_state_sha256=digest(c["P"]),
                  input_sha256=self.Xhash, subset_sha256={str(s): digest(D.E.RC.subset_idx(s)) for s in self.seeds},
                  data_sha256=c["cifar"].sha256))
            if self.expected_checkpoint:
                st = self.expected_checkpoint
                ext = st["extension_state"]
                if ext is None or int(ext["count"]) != c["spt"]:
                    raise ValueError("CHECK_FAILED: incomplete checkpoint diagnostic state")
                # A task checkpoint precedes sidecar writes. Recover the latter
                # deterministically, including exact accumulated update metrics.
                t = st["t"]
                self.rows = [r for r in self.rows if int(r["task"]) < t]
                self.streams = [r for r in self.streams if int(r["task"]) < t]
                if self.accum is None:
                    self.accum = ext["accum"].to(c["device"]).clone()
                    self.n = ext["count"].to(c["device"]).clone()
                else:
                    self.accum.copy_(ext["accum"]); self.n.copy_(ext["count"])
                self.prev_rng = ext["previous_rng"]
                ys = []
                for s in self.seeds:
                    generator = torch.Generator(); generator.set_state(self.prev_rng["g_lab"][s])
                    ys.append(D.E.RC.task_labels(generator))
                recovered = dict(c, t=t, Y=torch.stack(ys).to(c["device"]))
                self("task_end", recovered)
            if self.accum is not None:
                self.accum.zero_(); self.n.zero_()  # erase CUDA warmup/capture side effects
        elif event == "task_end":
            if c["diverged"]:
                C.put(self.out / "failure.json", dict(status="DIVERGED", failures=c["diverged"]))
                raise ValueError("DIVERGED; original checkpoint retained")
            if digest(c["X"]) != self.Xhash:
                raise ValueError("CHECK_FAILED: inputs changed")
            for i, initial in self.freeze.items():
                if (c["P"][i].requires_grad or not torch.equal(c["P"][i], initial) or
                    torch.count_nonzero(c["adam_m"][i]) or torch.count_nonzero(c["adam_v"][i])):
                    raise ValueError("CHECK_FAILED: frozen bias/gradient/moments")
            # Independently replay every draw, hash labels and all epoch orders.
            stream = dict(task=c["t"], seeds={})
            for s in self.seeds:
                gl, gb = torch.Generator(), torch.Generator()
                gl.set_state(self.prev_rng["g_lab"][s]); gb.set_state(self.prev_rng["g_batch"][s])
                label = D.E.RC.task_labels(gl)
                h = hashlib.sha256()
                for _ in range(c["epochs"]):
                    h.update(torch.randperm(D.E.N_IMAGES, generator=gb).numpy().tobytes())
                if (not torch.equal(gl.get_state(), c["g_lab"][s].get_state()) or
                    not torch.equal(gb.get_state(), c["g_batch"][s].get_state()) or
                    not torch.equal(label.to(c["Y"].device), c["Y"][self.seeds.index(s)])):
                    raise ValueError("CHECK_FAILED: label/batch stream")
                stream["seeds"][str(s)] = dict(label_sha256=digest(label), batch_sha256=h.hexdigest())
                self.prev_rng["g_lab"][s] = gl.get_state(); self.prev_rng["g_batch"][s] = gb.get_state()
            self.streams.append(stream)
            C.put(self.out / "streams.json", self.streams)
            before = digest(dict(P=c["P"], m=c["adam_m"], v=c["adam_v"], act=c["act"].state(), rng=self.prev_rng))
            data = units(c["P"], c["X"], c["act"])
            after = digest(dict(P=c["P"], m=c["adam_m"], v=c["adam_v"], act=c["act"].state(), rng=self.prev_rng))
            if before != after:
                raise ValueError("CHECK_FAILED: diagnostics mutated training state")
            data["adam_update_mean"] = self.accum.cpu().numpy() / int(self.n)
            data["adam_update_count"] = np.asarray(int(self.n))
            if int(self.n) != c["spt"]:
                raise ValueError("CHECK_FAILED: graph warmup entered diagnostic counts")
            path = self.out / "diagnostic_units" / f"t{c['t']:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **data)
            for slot, seed in enumerate(self.seeds):
                row = dict(arm=self.arm, seed=seed, task=c["t"])
                for layer in (1, 2):
                    for k in ("b", "m", "s", "r", "derivative_mean", "derivative_var", "dead", "w_norm"):
                        arr = data[f"l{layer}_{k}"][slot]
                        finite = arr[np.isfinite(arr)]
                        row[f"l{layer}_{k}_mean_finite"] = float(finite.mean()) if finite.size else ""
                        row[f"l{layer}_{k}_median_finite"] = float(np.median(finite)) if finite.size else ""
                        row[f"l{layer}_{k}_nonfinite"] = int((~np.isfinite(arr)).sum())
                    b = data[f"l{layer}_b"][slot]
                    row[f"l{layer}_bias_absmean"] = float(np.abs(b).mean())
                    sd = data[f"l{layer}_s"][slot]
                    row[f"l{layer}_s_zero"] = int((sd == 0).sum())
                    r = data[f"l{layer}_r"][slot]
                    row[f"l{layer}_r_posinf"] = int(np.isposinf(r).sum())
                    row[f"l{layer}_r_neginf"] = int(np.isneginf(r).sum())
                    row[f"l{layer}_r_nan"] = int(np.isnan(r).sum())
                row["l1_mean_minus_bias_absmax"] = float(np.abs(data["l1_m"][slot]-data["l1_b"][slot]).max())
                for layer in (1, 2, 3):
                    for k, name in enumerate(("adam_norm", "wd_norm", "actual_norm", "relative_actual", "zero_norm_fraction")):
                        row[f"l{layer}_{name}"] = float(data["adam_update_mean"][slot, layer-1, k])
                self.rows.append(row)
            C.write_rows(self.out / "diagnostics.csv", self.rows)
            self.accum.zero_(); self.n.zero_()
            C.put(self.out / "progress.json", dict(task=c["t"], at=C.utc(), diagnostic_sha256=C.sha(path)))
            return (C.OUT / "STOP").exists()
        return False

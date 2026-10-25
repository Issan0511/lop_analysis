"""Ordinary autograd training; no backward hooks, frozen parameters, or extra loss.

Run from repository root with --out raw/nonsinking_ce_1026.
Every member of a vectorized group is an independent network. Summing its
per-network mean CE preserves each network's ordinary gradient scale.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
TOL = 1e-12


def forward(params, x, multiclass=False):
    w, b, v, c = params
    z = torch.einsum("rnd,rhd->rnh", x, w) + b[:, None, :]
    h = F.relu(z)
    if multiclass:
        logits = torch.einsum("rnh,rhc->rnc", h, v) + c[:, None, :]
    else:
        logits = (h * v[:, None, :]).sum(-1) + c[:, None]
    return z, logits


def losses(logits, y, multiclass=False):
    if multiclass:
        return F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(y.shape).mean(1)
    return F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1)


@torch.no_grad()
def measure(params, x, y, multiclass=False):
    z, logits = forward(params, x, multiclass)
    mu = z.mean(1)
    sigma = z.std(1, correction=0)
    pred = logits.argmax(-1) if multiclass else (logits >= 0)
    return dict(z=z, loss=losses(logits, y, multiclass), mu=mu, sigma=sigma,
                ratio=mu / sigma, active=(z > 0).double().mean(1),
                accuracy=(pred == y).double().mean(1))


def numpy_gradients(params, x, y, multiclass=False):
    """Independent analytic CE gradient, including both layers and both biases."""
    w, b, v, c = [p.detach().numpy() for p in params]
    xn, yn = x.numpy(), y.numpy()
    z = np.einsum("rnd,rhd->rnh", xn, w) + b[:, None, :]
    h = np.maximum(z, 0)
    if multiclass:
        f = np.einsum("rnh,rhc->rnc", h, v) + c[:, None, :]
        ex = np.exp(f - f.max(-1, keepdims=True))
        error = ex / ex.sum(-1, keepdims=True)
        np.put_along_axis(error, yn[..., None], np.take_along_axis(error, yn[..., None], -1) - 1, -1)
        error /= xn.shape[1]
        gv = np.einsum("rnh,rnc->rhc", h, error)
        gz = np.einsum("rnc,rhc->rnh", error, v) * (z > 0)
        gc = error.sum(1)
    else:
        f = (h * v[:, None, :]).sum(-1) + c[:, None]
        prob = np.exp(-np.logaddexp(0, -f))
        error = (prob - yn) / xn.shape[1]
        gv = np.einsum("rn,rnh->rh", error, h)
        gz = error[:, :, None] * v[:, None, :] * (z > 0)
        gc = error.sum(1)
    return [np.einsum("rnh,rnd->rhd", gz, xn), gz.sum(1), gv, gc]


def two_point_certificate(params, x):
    z = forward(params, x)[0].detach().numpy()
    xn = x.numpy()
    k = np.einsum("rnd,rmd->rnm", xn, xn) + 1
    idx = z.argmax(1)
    r, h = np.indices(idx.shape)
    a, b = z[r, idx, h], z[r, 1 - idx, h]
    kaa, kab = k[r, idx, idx], k[r, idx, 1 - idx]
    v = params[2].detach().numpy()
    return dict(I=a * kab - b * kaa, kaa=kaa, kab=kab, a=a, b=b,
                conditions=(a > 0) & (b < 0) & (kab <= 0)
                & (kaa + kab > 0) & (a * kab - b * kaa > 0)
                & (v * (2 * idx - 1) > 0))


def config(name, x, w, b, v, c, steps=20000, lr=0.05, **kw):
    x = np.asarray(x, dtype=float)
    runs, n = x.shape[:2]
    y = np.broadcast_to(np.arange(n) >= n // 2, (runs, n)).astype(float).copy()
    return dict(name=name, x=x, y=y, w=np.asarray(w, float), b=np.asarray(b, float),
                v=np.asarray(v, float), c=np.asarray(c, float), steps=steps, lr=lr, **kw)


def make_configs():
    base = lambda name, a=2., b=-.25, v=.5, c=0., **kw: config(
        name, [[[-a], [a]]], [[[-.75], [.75]]], [[b, b]], [[-v, v]], [c], **kw)
    yield base("primary", steps=100000, certificate=True, recurrence=True)
    cases = [base("initial_errors", c=2.), base("zero_cross_gram", a=1.),
             base("positive_cross_gram", a=.5), base("high_initial_ratio", b=1.),
             base("wrong_readout_small", v=-.5), base("wrong_readout_large", v=-.75),
             config("biased_geometry", [[[-1.5], [4.5]]], [[[-1.1 / 6], [.75]]],
                    [[.725, -.25]], [[-.5, .5]], [0.])]
    for case in cases:
        yield case
    xs, ws, bs, vs, cs = [], [], [], [], []
    for seed in range(32):
        rng = np.random.default_rng(seed)
        xs.append(np.array([[-2., 0.], [2., 0.]]) + rng.uniform(-.1, .1, (2, 2)))
        ws.append(np.array([[-.75, 0.], [.75, 0.]]) + rng.uniform(-.05, .05, (2, 2)))
        bs.append(-.25 + rng.uniform(-.05, .05, 2))
        vs.append(np.array([-1, 1]) * (.5 + rng.uniform(-.1, .1, 2)))
        cs.append(rng.uniform(-2, 2))
    yield config("open_set", xs, ws, bs, vs, cs, certificate=True)
    xs, ws, bs, vs, cs = [], [], [], [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        positive = np.stack([np.linspace(1.8, 2.2, 32), rng.uniform(-.05, .05, 32)], -1)
        xs.append(np.concatenate([-positive, positive]))
        ws.append(np.array([[-.75, 0.], [.75, 0.]]) + rng.uniform(-.05, .05, (2, 2)))
        bs.append(-.25 + rng.uniform(-.05, .05, 2))
        vs.append(np.array([-1, 1]) * (.5 + rng.uniform(-.1, .1, 2)))
        cs.append(rng.uniform(-2, 2))
    yield config("within_class_spread", xs, ws, bs, vs, cs)
    ws, bs, vs, cs = [], [], [], []
    signs = np.r_[-np.ones(16), np.ones(16)]
    for seed in range(8):
        rng = np.random.default_rng(seed)
        ws.append((signs * (.75 + rng.uniform(-.05, .05, 32)))[:, None])
        bs.append(-.25 + rng.uniform(-.05, .05, 32))
        vs.append(signs * (.1 + rng.uniform(-.02, .02, 32)))
        cs.append(rng.uniform(-2, 2))
    yield config("width32", np.tile([[[-2.], [2.]]], (8, 1, 1)), ws, bs, vs, cs, certificate=True)
    yield base("minibatch_sgd", minibatch=True)
    yield base("adam", optimizer="Adam", lr=.01)
    classes = 10
    x = np.sqrt(2 * classes) * (np.eye(classes) - 1 / classes)
    cfg = config("softmax10", x[None], (.1 * x)[None], np.full((1, classes), -.25),
                 (.5 * np.eye(classes))[None], np.zeros((1, classes)), multiclass=True,
                 recurrence=True)
    cfg["y"] = np.arange(classes)[None]
    yield cfg


def train(cfg, out):
    started = time.time()
    name, steps = cfg["name"], cfg["steps"]
    mc = cfg.get("multiclass", False)
    x = torch.tensor(cfg["x"])
    y = torch.tensor(cfg["y"], dtype=torch.long if mc else torch.float64)
    params = [torch.nn.Parameter(torch.tensor(cfg[k])) for k in ("w", "b", "v", "c")]
    assert all(p.requires_grad for p in params)
    opt_class = torch.optim.Adam if cfg.get("optimizer") == "Adam" else torch.optim.SGD
    optimizer = opt_class(params, lr=cfg["lr"])
    initial = measure(params, x, y, mc)
    previous = initial
    initial_params = [p.detach().clone() for p in params]
    runs, units = initial["mu"].shape
    counters = {k: torch.zeros((runs, units), dtype=torch.int64) for k in ("mu", "ratio")}
    minima = {k: torch.full((runs, units), float("inf")) for k in ("mu", "ratio")}
    loss_up = torch.zeros(runs, dtype=torch.int64)
    loss_max_delta = torch.full((runs,), -float("inf"))
    gate_changes = torch.zeros(runs, dtype=torch.int64)
    grad_error = 0.
    recurrence_error = 0.
    invariant_error = 0.
    certificate = two_point_certificate(params, x) if cfg.get("certificate") else None
    if certificate is not None:
        assert certificate["conditions"].all(), (name, "initial theorem condition failed")
    observations = set([0, steps] + np.unique(np.geomspace(1, steps, 180).astype(int)).tolist()
                       + list(range(1000, steps + 1, 1000)))
    audits = {0, steps // 2, steps - 1}
    trace = {k: [] for k in ("step", "loss", "mu", "sigma", "ratio", "active", "accuracy", "w", "b", "v", "c")}

    def record(step, measured):
        trace["step"].append(step)
        for key in ("loss", "mu", "sigma", "ratio", "active", "accuracy"):
            trace[key].append(measured[key].detach().numpy().copy())
        for key, param in zip(("w", "b", "v", "c"), params):
            trace[key].append(param.detach().numpy().copy())

    record(0, initial)
    rng = np.random.default_rng(0)
    for step in range(steps):
        if cfg.get("minibatch"):
            idx = int(rng.integers(x.shape[1]))
            xt, yt = x[:, idx:idx + 1], y[:, idx:idx + 1]
        else:
            xt, yt = x, y
        optimizer.zero_grad(set_to_none=True)
        _, logits = forward(params, xt, mc)
        loss = losses(logits, yt, mc).sum()
        loss.backward()
        if step in audits:
            analytical = numpy_gradients(params, xt, yt, mc)
            err = max(np.max(np.abs(p.grad.numpy() - ref)) for p, ref in zip(params, analytical))
            grad_error = max(grad_error, float(err))
            assert err < 1e-10, (name, step, err)
        if cfg.get("recurrence") and step in audits:
            z0 = previous["z"].numpy()
            vv = params[2].detach().numpy()
            if mc:
                classes = x.shape[1]
                a0, b0 = z0[0, 0, 0], z0[0, 1, 0]
                d0 = vv[0, 0, 0] - vv[0, 0, 1]
                q = (classes - 1) / (np.exp(a0 * d0) + classes - 1)
                pred = [a0 + cfg["lr"] * (2 * classes - 1) * q * d0 / classes,
                        b0 - cfg["lr"] * q * d0 / classes,
                        d0 + cfg["lr"] * a0 * q / (classes - 1)]
            else:
                a0, u0 = z0[0, 0, 0], abs(vv[0, 0])
                q = np.exp(-np.logaddexp(0, a0 * u0))
                pred = [a0 + 5 * cfg["lr"] * u0 * q / 2,
                        u0 + cfg["lr"] * a0 * q / 2]
        optimizer.step()
        measured = measure(params, x, y, mc)
        assert all(torch.isfinite(measured[k]).all() for k in ("loss", "mu", "sigma", "ratio"))
        for key in ("mu", "ratio"):
            delta = measured[key] - previous[key]
            counters[key] += delta < -TOL
            minima[key] = torch.minimum(minima[key], delta)
        delta = measured["loss"] - previous["loss"]
        loss_up += delta > TOL
        loss_max_delta = torch.maximum(loss_max_delta, delta)
        gate_changes += ((measured["z"] > 0) != (initial["z"] > 0)).flatten(1).sum(1)
        if cfg.get("recurrence") and step in audits:
            z1 = measured["z"].numpy()
            vv = params[2].detach().numpy()
            actual = ([z1[0, 0, 0], z1[0, 1, 0], vv[0, 0, 0] - vv[0, 0, 1]] if mc
                      else [z1[0, 0, 0], abs(vv[0, 0])])
            err = float(np.max(np.abs(np.array(actual) - pred)))
            recurrence_error = max(recurrence_error, err)
            assert err < 1e-10, (name, "recurrence", step, err)
        if certificate is not None and step + 1 in observations:
            new_cert = two_point_certificate(params, x)
            invariant_error = max(invariant_error, float(np.max(np.abs(new_cert["I"] - certificate["I"]))))
            assert new_cert["conditions"].all(), (name, "invariant set exit", step)
        if step + 1 in observations:
            record(step + 1, measured)
        previous = measured
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **{k: np.asarray(v) for k, v in trace.items()},
                        x=cfg["x"], y=cfg["y"])
    rows = []
    for run in range(runs):
        row = dict(group=name, seed=run, steps=steps, units=units, samples=x.shape[1],
                   lr=cfg["lr"], optimizer=opt_class.__name__, minibatch=int(cfg.get("minibatch", False)),
                   loss_initial=float(initial["loss"][run]), loss_final=float(previous["loss"][run]),
                   accuracy_initial=float(initial["accuracy"][run]), accuracy_final=float(previous["accuracy"][run]),
                   loss_increasing_steps=int(loss_up[run]), max_loss_step=float(loss_max_delta[run]),
                   gate_changed_observations=int(gate_changes[run]), gradient_max_error=grad_error,
                   recurrence_max_error=recurrence_error, invariant_max_error=invariant_error,
                   initial_certificate=int(certificate is not None), elapsed_seconds=time.time() - started)
        for key in ("mu", "sigma", "ratio", "active"):
            for endpoint, m in (("initial", initial), ("final", previous)):
                vals = m[key][run].numpy()
                row[f"{key}_{endpoint}_mean"] = float(vals.mean())
                row[f"{key}_{endpoint}_units"] = json.dumps(vals.tolist())
        for key in ("mu", "ratio"):
            row[f"{key}_down_unit_steps"] = int(counters[key][run].sum())
            row[f"{key}_down_units"] = int((counters[key][run] > 0).sum())
            row[f"{key}_minimum_step"] = float(minima[key][run].min())
        for key, before, after in zip(("w", "b", "v", "c"), initial_params, params):
            row[f"{key}_total_displacement"] = float(torch.linalg.vector_norm((after[run].detach() - before[run]).flatten()))
        rows.append(row)
    print(json.dumps(dict(group=name, runs=runs, seconds=round(time.time() - started, 2),
                          final_loss_range=[min(r["loss_final"] for r in rows), max(r["loss_final"] for r in rows)],
                          mu_down=sum(r["mu_down_unit_steps"] for r in rows),
                          ratio_down=sum(r["ratio_down_unit_steps"] for r in rows)), ensure_ascii=False), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("raw/nonsinking_ce_1026"))
    parser.add_argument("--results", type=Path, default=Path("results/nonsinking_ce_1026"))
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in make_configs() if not args.only or c["name"] in args.only]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    provenance = dict(git_hash=head, torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                      dtype="float64", device="CPU", loss="standard mean BCEWithLogits / softmax CE", tolerance=TOL,
                      all_parameters_trained=True, gradient_hooks=False, regularization=False,
                      configurations=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in c.items()} for c in cfgs])
    (args.results / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = []
    for cfg in cfgs:
        rows.extend(train(cfg, args.out))
        with (args.results / "verdict.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = [dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted(args.out.glob("*.npz"))]
    (args.results / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

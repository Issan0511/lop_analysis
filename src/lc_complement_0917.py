"""LC complement C1/C2/C3/C4. See specs/spec_lc_complement_0917.md.

No changes to the imported experiment engines. CPU commands use one thread;
the optional C1 GPU replay retains the original batched engine and its streams.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src import resp_ee_0917 as R

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "lc_complement_0917"
SPEC = "specs/spec_lc_complement_0917.md"
ARCHIVE = Path.home() / "Projects/obsidian-research-data/resp_ee_0917/results/resp_ee_0917/runs"
REFERENCE = ROOT / "results/resp_ee_0917/runs"
SHOCK_ARMS = ("N", "K_rev", "K_mid2", "K_sw2", "K_hold2", "K_hold2_L2")
C3_ARMS = {"E1": (1., 1., 1e-4), "S36": (1., 3.6, 1e-4),
           "C36": (3.6, 1., 1e-4), "E36": (3.6, 3.6, 1e-4),
           "E1_lr1e3": (1., 1., 1e-3)}  # cap c, slope ck, lr


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def code_hashes(part):
    paths = [Path(__file__), Path(R.__file__), Path(R.EL.__file__), Path(R.EG.__file__),
             Path(R.H.__file__), Path(R.RL.__file__), Path(R.SH.__file__),
             ROOT / f"analysis/{EXPERIMENT}/verdict.py"]
    if part == "c1-gpu":
        paths += [ROOT / f"analysis/{EXPERIMENT}/gpu.py", ROOT / "src/neff_pred_0917.py",
                  ROOT / "src/relu_gelu_silu_rl_0914.py", ROOT / "src/layer_chimera_rl_0914.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def provenance(out, part, smoke=False, **config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {out}")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    dirty = git("status", "--porcelain", "--", "src", f"analysis/{EXPERIMENT}", SPEC)
    if not smoke and dirty:
        raise RuntimeError("Commit code and specification before collecting main data")
    p = dict(experiment=EXPERIMENT, part=part, status="RUNNING", smoke=smoke,
             git_hash=git("rev-parse", "HEAD"), git_dirty_code=dirty,
             registration_commit=git("log", "-1", "--format=%H", "--", SPEC),
             spec_sha256=sha(ROOT / SPEC), torch=torch.__version__,
             cpu_capability=torch.backends.cpu.get_cpu_capability(),
             threads=torch.get_num_threads(), flush_denormal=R._flush_is_on(),
             hostname=platform.node(), config=config,
             code_sha256=code_hashes(part))
    dump(out / "provenance.json", p)
    return p


def complete(out, p, **extra):
    p.update(status="COMPLETE", **extra)
    dump(Path(out) / "provenance.json", p)


def clone_state(ck):
    return ([v.detach().clone().requires_grad_(True) for v in ck["params"]],
            ([v.clone() for v in ck["m"]], [v.clone() for v in ck["v"]], [ck["tc"]]))


def load_branches(seed, archive=ARCHIVE, reference=REFERENCE, mnist=None):
    """Verify archive manifest, state hashes, source code, dataset and CPU lineage."""
    folder = Path(archive) / f"s{seed}"
    ref = Path(reference) / f"s{seed}"
    src = json.loads((ref / "provenance.json").read_text())
    state_path = folder / "branch_states.pt"
    manifest = json.loads((ROOT / "results/resp_ee_0917/backup_manifest.json").read_text())
    # The archive's manifest has a list under 'files'; reject unlisted or altered input.
    entries = manifest if isinstance(manifest, list) else manifest.get("files", [])
    match = [e for e in entries if e.get("backup") == str(state_path)]
    if len(match) != 1 or sha(state_path) != match[0]["sha256"]:
        raise ValueError(f"Checkpoint missing from manifest or hash mismatch: {state_path}")
    for key, current in (("cpu_capability", torch.backends.cpu.get_cpu_capability()),
                         ("torch", torch.__version__), ("threads", torch.get_num_threads()),
                         ("flush_denormal", R._flush_is_on())):
        if src[key] != current:
            raise ValueError(f"Source runtime mismatch: {key}: {src[key]} != {current}")
    for path, expected in src["code_sha256"].items():
        if sha(ROOT / path) != expected:
            # This host grew unrelated cap/diagnostic code after resp_ee.
            # resp_ee uses only forward2: require its AST to match the recorded blob.
            if path != "src/mucap_el_run_0916.py":
                raise ValueError(f"Source code mismatch: {path}")
            old = subprocess.check_output(["git", "show", f"{src['git_hash']}:{path}"], cwd=ROOT)
            if hashlib.sha256(old).hexdigest() != expected:
                raise ValueError("Recorded host blob does not match provenance")
            def forward_ast(source):
                return ast.dump(next(n for n in ast.parse(source).body
                                     if isinstance(n, ast.FunctionDef) and n.name == "forward2"))
            if forward_ast(old) != forward_ast((ROOT / path).read_text()):
                raise ValueError("Host forward2 has changed")
    if mnist is None:
        mnist = R.H.Mnist(torch.device("cpu"))
    if mnist.sha256 != src["data_sha256"]:
        raise ValueError("MNIST digest mismatch")
    cks = torch.load(state_path, map_location="cpu", weights_only=False)
    x = mnist.train_x[R.RL.subset_idx(seed)]
    for t, ck in cks.items():
        p, adam = clone_state(ck)
        if R.SH.state_sha256(p, adam, R.ACT) != ck["state_sha256"]:
            raise ValueError(f"Corrupt branch state t{t}")
        with torch.no_grad():
            z1, _, z2, _, _ = R.forward(p, x, None, None)
        if not torch.equal(z1, ck["z1"]) or not torch.equal(z2, ck["z2"]):
            raise ValueError(f"Branch forward differs from archived CPU trajectory t{t}")
    return cks, x, dict(checkpoint_sha256=sha(state_path),
                        source_provenance_sha256=sha(ref / "provenance.json"),
                        source_arms_sha256=sha(ref / "arms.csv"), data_sha256=mnist.sha256)


@torch.no_grad()
def confusion(a, g, family="ELU", alpha=1., batch=16):
    """Element masks, per-batch layer-wide sample std, and per-unit |g| n_eff.

    Return counts as well as conditional rates; a zero denominator is undefined.
    All-zero derivatives have n_eff=0. Negative GELU/SiLU lobes cannot cancel it.
    """
    # LC's diagnostic first transfers activations to CPU. This also avoids
    # synchronizing the GPU once for every small diagnostic batch.
    a, g = a.detach().cpu(), g.detach().cpu()
    if a.ndim != 2 or a.shape != g.shape or not torch.isfinite(a).all() or not torch.isfinite(g).all():
        raise ValueError("Expected matching finite [image, unit] fields")
    if batch < 1:
        raise ValueError("batch must be positive")
    if family not in ("ELU", "CELU", "GELU", "SILU", "R", "LR"):
        raise ValueError(f"Unknown activation family: {family}")
    lc = torch.empty_like(a, dtype=torch.bool)
    for start in range(0, len(a), batch):
        ab = a[start:start + batch]
        sigma = ab.std().item() + 1e-8
        band = .05 * sigma
        if family == "R":
            band = 1e-5
        elif family == "LR":
            band = max(band, 10 * .1 * sigma)  # original GPU engine's leak=0.1
        dead = ab.abs() < band
        if family in ("ELU", "CELU"):
            dead = dead | (ab < -alpha + .02)
        lc[start:start + batch] = dead
    tr, zero = g.abs() < 1e-3, g == 0
    ga = g.double().abs()
    den = len(a) * ga.square().sum(0)
    neff = torch.where(den > 0, ga.sum(0).square() / den.clamp_min(1e-300), 0.)
    counts = dict(n=a.numel(), lc_n=int(lc.sum()), tr_n=int(tr.sum()), zero_n=int(zero.sum()),
                  false_dead_n=int((lc & (g.abs() >= .1)).sum()),
                  missed_n=int((tr & ~lc).sum()), missed_zero_n=int((zero & ~lc).sum()))
    return {**counts, "lc_frac": counts["lc_n"] / a.numel(),
            "tr_frac": counts["tr_n"] / a.numel(), "zero_frac": counts["zero_n"] / a.numel(),
            "A": counts["false_dead_n"] / counts["lc_n"] if counts["lc_n"] else None,
            "B": counts["missed_n"] / counts["tr_n"] if counts["tr_n"] else None,
            "B_zero": counts["missed_zero_n"] / counts["zero_n"] if counts["zero_n"] else None,
            "neff_abs": float(neff.mean())}, neff.cpu().numpy()


def c1_ee(seed, out, archive=ARCHIVE, reference=REFERENCE, smoke=False):
    import pandas as pd
    out = Path(out)
    prov = provenance(out, "c1-ee", smoke, seed=seed)
    cks, x, sources = load_branches(seed, archive, reference)
    old = pd.read_csv(Path(reference) / f"s{seed}/arms.csv", float_precision="round_trip")
    rows, units = [], {}
    for name, arm in R.ARMS.items():
        ck = cks[arm["branch"]]
        sh = R.build_shift(arm, cks)
        with torch.no_grad():
            z1, a1, z2, a2, lg = R.forward(ck["params"], x, torch.arange(len(x)), R.make_forward(sh))
            natural_lg = R.forward(ck["params"], x, None, None)[4]
        if not torch.equal(lg, natural_lg):
            raise ValueError(f"Forward matching failed: {name}")
        erow = old[(old.arm == name) & (old.k == 1)]
        if len(erow) != 1 or not bool(erow.iloc[0].finite):
            raise ValueError(f"Invalid source endpoint: {name}")
        for layer, z, a in ((1, z1, a1), (2, z2, a2)):
            d = sh.get(layer) if sh else None
            g = R.dphi_train(z if d is None else z + d)
            stats, u = confusion(a, g)
            rows.append(dict(seed=seed, arm=name, branch=arm["branch"], layer=layer,
                             E=float(erow.iloc[0].online_acc), **stats))
            units[f"{name}_l{layer}_neff_abs"] = u
    R.write_csv(out / "rows.csv", rows)
    np.savez_compressed(out / "units.npz", **units)
    complete(out, prov, sources=sources)


def shock_gamma(arm, step, continuation=1, steps=6000):
    if arm not in SHOCK_ARMS or step < 0 or step >= steps or continuation not in (1, 2):
        raise ValueError("Invalid shock coordinate")
    if continuation == 2 or arm == "N":
        return 1., 1.
    gamma = 1.
    if arm == "K_rev":
        # Epochs 10,20,...,80 (1-based); each shock lasts one complete epoch.
        epoch = step // R.SPE
        if (epoch + 1) % 10 == 0:
            gamma = (1.5, .5, .25, 2.)[(epoch // 10) % 4]
    elif arm == "K_sw2" and step < R.SPE:
        gamma = 2.
    elif arm == "K_mid2" and steps // 2 <= step < steps // 2 + R.SPE:
        gamma = 2.
    elif arm in ("K_hold2", "K_hold2_L2"):
        gamma = 2.
    return (1. if arm == "K_hold2_L2" else gamma), gamma


class ShockForward:
    def __init__(self, arm, continuation, steps):
        self.arm, self.continuation, self.steps = arm, continuation, steps
        self.step = 0

    def __call__(self, p, x, ob):
        g1, g2 = shock_gamma(self.arm, self.step, self.continuation, self.steps)
        self.step += 1
        return shock_forward(p, x, g1, g2)


def shock_forward(p, x, g1=1., g2=1.):
    z1 = (x @ p[0].T + p[1]) * g1
    a1 = R.ACT.phi(z1)
    z2 = (a1 @ p[2].T + p[3]) * g2
    a2 = R.ACT.phi(z2)
    return z1, a1, z2, a2, a2 @ p[4].T + p[5]


def c2(seed, out, archive=ARCHIVE, reference=REFERENCE, smoke=False):
    import pandas as pd
    out = Path(out)
    epochs = 1 if smoke else 80
    prov = provenance(out, "c2", smoke, seed=seed, epochs=epochs, branches=[2, 5], arms=SHOCK_ARMS)
    mnist = R.H.Mnist(torch.device("cpu"))
    if smoke:
        prows, _, cks, x, sources = R.run_prefix(seed, mnist, epochs, 7, save_at=(2, 5))
        old = pd.DataFrame(prows)
    else:
        cks, x, sources = load_branches(seed, archive, reference, mnist)
        old = pd.read_csv(Path(reference) / f"s{seed}/prefix.csv", float_precision="round_trip")
    rows, curves = [], {}
    for branch in (2, 5):
        ck = cks[branch]
        for arm in SHOCK_ARMS:
            p, adam = clone_state(ck)
            gl, gb = R.gen_from(ck["g_lab"]), R.gen_from(ck["g_batch"])
            for k in (1, 2):
                y = R.RL.task_labels(gl)
                fwd = ShockForward(arm, k, R.SPE * epochs)
                ce, acc, stp, epsf, _ = R.train_task(p, adam, x, y, gb, epochs, fwd)
                if not torch.isfinite(ce).all():
                    raise FloatingPointError(f"Nonfinite C2 branch={branch} arm={arm} k={k}")
                state = R.SH.state_sha256(p, adam, R.ACT)
                if arm == "N":
                    ref = old[old.task == branch + k].iloc[0]
                    if state != ref.state_sha256 or float(acc.double().mean()) != ref.online_acc:
                        raise ValueError("Natural continuation does not match original trajectory")
                # Measure after removal as well as while the final training gamma is applied.
                with torch.no_grad():
                    obs = {}
                    for tag, gamma in (("off", (1., 1.)),
                                       ("on", shock_gamma(arm, R.SPE * epochs - 1, k, R.SPE * epochs))):
                        z1, a1, z2, a2, lg = shock_forward(p, x, *gamma)
                        obs[f"memo_{tag}"] = float((lg.argmax(1) == y).double().mean())
                        for layer, z, a, ga in ((1, z1, a1, gamma[0]), (2, z2, a2, gamma[1])):
                            stats, _ = confusion(a, ga * R.dphi_train(z))
                            obs.update({f"{tag}_l{layer}_{q}": stats[q] for q in
                                        ("lc_frac", "tr_frac", "zero_frac", "neff_abs")})
                rows.append(dict(seed=seed, branch=branch, arm=arm, k=k, task=branch + k,
                                 state_sha256=state, **R.task_row(ce, acc, stp, epsf, y, R.SPE * epochs), **obs))
                curves[f"t{branch}_{arm}_k{k}_acc"] = acc.numpy()
                curves[f"t{branch}_{arm}_k{k}_ce"] = ce.numpy()
            print(f"C2 seed={seed} branch={branch} arm={arm} complete", flush=True)
    R.write_csv(out / "rows.csv", rows)
    np.savez_compressed(out / "curves.npz", **curves)
    complete(out, prov, sources=sources)


def c3_phi(z, cap, slope):
    # Native ELU/CELU kernels avoid the artificial expm1-autograd floor.
    if slope == cap:
        return F.elu(z, alpha=cap)
    if slope == 1.:
        return F.celu(z, alpha=cap)
    return torch.where(z > 0, z, F.elu(z * (slope / cap), alpha=cap))


def c3_forward(p, x, cap, slope):
    z1 = F.linear(x, p[0], p[1])
    a1 = c3_phi(z1, cap, slope)
    z2 = F.linear(a1, p[2], p[3])
    a2 = c3_phi(z2, cap, slope)
    return z1, a1, z2, a2, F.linear(a2, p[4], p[5])


def c3_init(seed):
    # LC benchmark RNG order: 50 tasks' train/validation labels, then nn.Linear.
    torch.manual_seed(seed)
    labels = []
    for _ in range(50):
        labels.append(torch.randint(10, (1200,)))
        torch.randint(10, (500,))
    layers = [torch.nn.Linear(a, b) for a, b in zip((784, 100, 100), (100, 100, 10))]
    return [p for layer in layers for p in layer.parameters()], labels


def c3(seed, arm, out, smoke=False):
    out = Path(out)
    cap, slope, lr = C3_ARMS[arm]
    tasks, epochs = (2, 1) if smoke else (50, 400)
    prov = provenance(out, "c3", smoke, seed=seed, arm=arm, cap=cap, slope=slope,
                      lr=lr, tasks=tasks, epochs=epochs, protocol="LC-first1200-fixed-order-native-ELU")
    mnist = R.H.Mnist(torch.device("cpu"))
    x = mnist.train_x[:1200]
    p, labels = c3_init(seed)
    opt = torch.optim.Adam(p, lr=lr)
    rows, units = [], {}
    for task in range(1, tasks + 1):
        y = labels[task - 1]
        correct = 0
        for _ in range(epochs):
            for start in range(0, 1200, 16):
                lg = c3_forward(p, x[start:start+16], cap, slope)[4]
                loss = F.cross_entropy(lg, y[start:start+16])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"C3 nonfinite loss task={task}")
                correct += int((lg.detach().argmax(1) == y[start:start+16]).sum())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            z1, a1, z2, a2, lg = c3_forward(p, x, cap, slope)
        floor = float(torch.bincount(y, minlength=10).max()) / 1200
        online = correct / (1200 * epochs)
        row = dict(seed=seed, arm=arm, task=task, online_acc=online, floor=floor,
                   fit=(online - floor) / (1 - floor), memo=float((lg.argmax(1) == y).double().mean()),
                   mu2_norm=float(a1.double().mean(0).norm()))
        for layer, z, a in ((1, z1, a1), (2, z2, a2)):
            zz = z.detach().requires_grad_(True)
            g = torch.autograd.grad(c3_phi(zz, cap, slope).sum(), zz)[0]
            stats, neff = confusion(a, g, alpha=cap)
            row.update({f"l{layer}_{key}": value for key, value in stats.items()})
            row[f"zbar_l{layer}"] = float(z.double().mean())
            units.setdefault(f"neff_l{layer}", []).append(neff)
        rows.append(row)
        R.write_csv(out / "rows.csv", rows)
        # Last completed task checkpoint is replaced atomically, never mixed with CSV rows.
        tmp = out / "checkpoint.tmp"
        torch.save(dict(task=task, params=[q.detach() for q in p], optimizer=opt.state_dict(),
                        labels=labels, config=prov["config"]), tmp)
        tmp.replace(out / "checkpoint.pt")
        print(f"C3 {arm} seed={seed} task={task}/{tasks} online={online:.4f}", flush=True)
    np.savez_compressed(out / "units.npz", **{k: np.stack(v) for k, v in units.items()})
    complete(out, prov, data_sha256=mnist.sha256,
             labels_sha256=hashlib.sha256(torch.stack(labels).numpy().tobytes()).hexdigest())


def adam_bound(beta1=.9, beta2=.999, step=None):
    """Sharp Cauchy-Schwarz bound on |m_hat|/sqrt(v_hat), for zero initial moments.

    Epsilon only reduces the step. The uniform bound requires beta1^2 < beta2.
    Persistent moments must use the total Adam age, not the continuation age.
    """
    if not 0 <= beta1 < 1 or not 0 < beta2 < 1 or beta1**2 >= beta2:
        raise ValueError("Require 0<=beta1<1, 0<beta2<1, beta1^2<beta2")
    r = beta1**2 / beta2
    bound = (1-beta1) / math.sqrt((1-beta2) * (1-r))
    if step is None:
        return bound
    if not isinstance(step, int) or step < 1:
        raise ValueError("Adam step must be a positive integer")
    return bound * math.sqrt((1-r**step)*(1-beta2**step)) / (1-beta1**step)


def c4(out):
    out = Path(out)
    prov = provenance(out, "c4")
    coeff = adam_bound()
    per_step = coeff * 1e-3 * (104 + 1)
    result = dict(uniform_adam_factor=coeff, eta=.001, mu_l1=104., depth=16.,
                  max_mean_z_step=per_step, necessary_steps=math.ceil(16 / per_step),
                  interpretation="LOWER_BOUND_ONLY_NOT_RECOVERY_GUARANTEE",
                  assumptions="zero-initialized Adam; fixed input mean; no external shock; epsilon>=0")
    dump(out / "bound.json", result)
    R.write_csv(out / "verdict.csv", [result])
    (out / "summary.md").write_text(
        "# C4: recovery time is bounded below, not above\n\n"
        "Cauchy–Schwarz gives |Δθ| ≤ η (1−β1)/sqrt((1−β2)(1−β1²/β2)). "
        f"For (.9,.999) the factor is {coeff:.6f}, not 3.16 for arbitrary gradient histories. "
        f"With fixed ||μ||₁=104 and η=.001, depth 16 requires at least {result['necessary_steps']} updates. "
        "This necessary condition gives no finite upper bound on recovery time. "
        "Zero gradients, unfavorable directions, or persistent transport can prevent recovery. "
        "For layer 2 with moving inputs, Δz̄=Δw·μ+w·Δμ+Δw·Δμ+Δb; "
        "the bound above controls only Δw·μ+Δb.\n")
    complete(out, prov)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part", choices=("c1-ee", "c1-gpu", "c2", "c3", "c4"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=C3_ARMS, default="E1")
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke and args.part not in ("c1-ee", "c2", "c3"):
        ap.error("--smoke is supported only by c1-ee, c2 and c3")
    torch.set_num_threads(1)
    torch.set_flush_denormal(args.part != "c3")
    R.H.setup("cpu")
    if args.part == "c1-ee":
        c1_ee(args.seed, args.out, args.archive, args.reference, args.smoke)
    elif args.part == "c1-gpu":
        from analysis.lc_complement_0917.gpu import run
        run(args.out)
    elif args.part == "c2":
        c2(args.seed, args.out, args.archive, args.reference, args.smoke)
    elif args.part == "c3":
        c3(args.seed, args.arm, args.out, args.smoke)
    else:
        c4(args.out)


if __name__ == "__main__":
    main()

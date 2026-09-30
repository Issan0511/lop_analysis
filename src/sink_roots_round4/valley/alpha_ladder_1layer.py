"""Scaled-down ELU ladder in the one-hidden-layer RL-MNIST box (same box as pplus_sign_1layer_0925 v3long).

phi_alpha(z) = z (z > 0), alpha * (exp(z/alpha) - 1) (z <= 0)   = alpha * ELU(z/alpha)  (CELU(alpha))
phi'_alpha(z) = 1 (z > 0), exp(z/alpha) (z <= 0).  alpha -> 0 gives ReLU.  No valley for any alpha > 0.

Two derivative implementations (float32 training):
  m1 : autograd of alpha*expm1(z/alpha)  -> derivative fl(expm1(u))+1, exactly 0 for u = z/alpha < -16.6355 (CPU).
       alpha = 1 is bit-identical to the ELU arm of v3long (same ops, multiply/divide by 1.0).
  ex : custom autograd, derivative exp(z/alpha) (float32 underflow only below u ~ -103).
Other activations (R, GELU, SILU, LR) as in v3long.

Box: MNIST raw/255, subset N=1200 (pmnist_0905 hash), 784-100-10, init U(+-1/sqrt(din)), Adam 1e-3 (.9/.999, eps 1e-8,
state carried across tasks), batch 16, 4000 updates/task, random labels each task.  CPU, 1 thread.
Records everything v3long records, plus per-unit task displacement |dw_i|, the top input's demand v_i.e_top at the
switch, and per-input z snapshots (switch / +200 / end) for tasks in SNAP.
"""
import argparse, gzip, hashlib, math, time
from pathlib import Path
import numpy as np, torch

DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
N, BATCH, H, K = 1200, 16, 100, 10
KS = (1, 10, 50, 200, 1000)
SNAP = (2, 10, 30, 50, 100, 150, 200)   # sink_roots_0930 round 2: --snap_every adds multiples
SQRT2 = math.sqrt(2.0); INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


class ELUAExact(torch.autograd.Function):
    """alpha*ELU(z/alpha) with the exact derivative exp(z/alpha) on the negative side."""
    @staticmethod
    def forward(ctx, z, alpha):
        ctx.save_for_backward(z); ctx.alpha = alpha
        return torch.where(z > 0, z, alpha * torch.expm1((z / alpha).clamp_max(0)))

    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        d = torch.where(z > 0, torch.ones_like(z), torch.exp((z / ctx.alpha).clamp_max(0)))
        return g * d, None


def make_act(act, alpha, dmode):
    if act == "ELUA":
        if dmode == "m1":
            return lambda z: torch.where(z > 0, z, alpha * torch.expm1((z / alpha).clamp_max(0)))
        if dmode == "ex":
            return lambda z: ELUAExact.apply(z, alpha)
        raise ValueError(dmode)
    if act == "LR":   return lambda z: torch.where(z > 0, z, z * 0.1)
    if act == "R":    return lambda z: torch.relu(z)
    if act == "GELU": return lambda z: z * 0.5 * (1.0 + torch.erf(z / SQRT2))
    if act == "SILU": return lambda z: z * torch.sigmoid(z)
    raise ValueError(act)


def make_gate(act, alpha):
    """float64 diagnostic phi'(z) (the mathematical derivative, not the float32 training one)."""
    if act == "ELUA": return lambda z: torch.where(z > 0, torch.ones_like(z), (z / alpha).clamp_max(0).exp())
    if act == "LR":   return lambda z: torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
    if act == "R":    return lambda z: (z > 0).to(z.dtype)
    if act == "GELU": return lambda z: 0.5 * (1.0 + torch.erf(z / SQRT2)) + z * torch.exp(-0.5 * z * z) * INV_SQRT_2PI
    if act == "SILU":
        def g(z):
            s = torch.sigmoid(z); return s * (1.0 + z * (1.0 - s))
        return g
    raise ValueError(act)


def initial(seed, dims=(784, H, K)):
    g = stream("init", seed); p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p


@torch.no_grad()
def unit_stats(W1, b1, X, gate):
    z = (X.double() @ W1.double().T + b1.double())   # [N,H] float64 diagnostics
    g = gate(z)
    top2 = z.topk(2, dim=0).values
    return dict(z=z, g=g, pplus=(z > 0).double().mean(0), gbar=g.mean(0), m=z.mean(0), s=z.std(0),
                zmax=top2[0], zmax2=top2[1], zmin=z.min(0).values)


@torch.no_grad()
def switch_preds(P, X, y_old, y_new, act64, gate, mu_hat, xmu):
    W1, b1, W2, b2 = [q.double() for q in P]
    st = unit_stats(W1, b1, X, gate); z, g = st["z"], st["g"]
    a = act64(z); logits = a @ W2.T + b2; p = logits.softmax(-1)
    V = W2.T; vbar = V.mean(1)
    v_yold = V[:, y_old].T
    P_onehot = -((v_yold - vbar[None, :]) * g * xmu[:, None]).sum(0)
    vp = p @ V.T
    P_exact = -((vp - vbar[None, :]) * g * xmu[:, None]).sum(0)
    openm = (z > 0).double()
    P_exact_open = -((vp - vbar[None, :]) * g * xmu[:, None] * openm).sum(0)
    e = p - torch.nn.functional.one_hot(y_new, K).to(p.dtype)
    ve = e @ V.T                                                            # [N,H] demand: <0 = wants more output
    G_new = -((ve * g) * xmu[:, None]).sum(0)
    top_idx = z.argmax(0)                                                   # [H]
    demand_top = ve[top_idx, torch.arange(H)]
    delta = (ve * g) / N
    grad_W1 = delta.T @ X.double()
    coh = (torch.sign(grad_W1) * mu_hat[None, :]).sum(1) / mu_hat.sum()
    return dict(pplus=st["pplus"], gbar=st["gbar"], m=st["m"], s=st["s"], wnorm=W1.norm(dim=1), zmax=st["zmax"],
                P_onehot=P_onehot, P_exact=P_exact, P_exact_open=P_exact_open, P_exact_closed=P_exact - P_exact_open,
                G_new=G_new, coh=coh, wmu=(W1 @ mu_hat), demand_top=demand_top, gradnorm=grad_W1.norm(dim=1),
                train_acc=(logits.argmax(-1) == y_old).double().mean(), z=z)


def run(act, alpha, dmode, seed, tasks, steps, lr, out, tag, lr1=None, lr2=None, eps1=None, eps2=None, init1=1.0, init2=1.0, freeze_bias=False,
        eps_diag=False):
    torch.manual_seed(seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32); del ax
    mu = X.double().mean(0); mu_hat = mu / mu.norm(); xmu = X.double() @ mu_hat
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P0 = initial(seed)
    P0[0] = P0[0] * init1; P0[1] = P0[1] * init1; P0[2] = P0[2] * init2
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in P0]
    if freeze_bias:
        P[1].requires_grad_(False)
    EPS = 1e-8
    g1 = {"params": [P[0]] + ([] if freeze_bias else [P[1]]), "lr": lr if lr1 is None else lr1, "eps": EPS if eps1 is None else eps1}
    g2 = {"params": [P[2]], "lr": lr if lr2 is None else lr2, "eps": EPS if eps2 is None else eps2}
    g3 = {"params": [P[3]], "lr": lr, "eps": EPS}
    optim = torch.optim.Adam([g1, g2, g3], lr=lr, betas=(0.9, 0.999), eps=EPS)
    actf = make_act(act, alpha, dmode); gate = make_gate(act, alpha)
    act64 = make_act(act, alpha, "m1") if act == "ELUA" else actf         # float64 forward for diagnostics
    y_old = None
    per_task = {k: [] for k in ("pplus", "gbar", "m", "s", "wnorm", "train_acc", "train_ce", "bias", "zmax", "zmax2", "zmin", "disp", "dbias")}
    diag = {"v0_m_nonzero_max": [], "v0_m_nonzero_end": [], "max_abs_dtheta": []}   # sink_roots_0930 round 4 R_eps_zero_v2
    swk = ("pplus", "gbar", "m", "s", "wnorm", "P_onehot", "P_exact", "P_exact_open", "P_exact_closed", "G_new", "coh", "wmu", "zmax",
           "demand_top", "gradnorm")
    sw = {k: [] for k in swk + ("train_acc", "zmax_k", "d_wmu", "d_b", "pplus_k", "rho_k", "m_k")}
    snaps = {"z_sw": [], "z_200": [], "z_end": []}
    t0 = time.monotonic()
    for task in range(1, tasks + 1):
        y_new = torch.randint(K, (N,), generator=glabel)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:steps * BATCH].reshape(steps, BATCH)
        W_start = P[0].detach().clone(); b_start = P[1].detach().clone()
        if y_old is not None:
            pr = switch_preds([q.detach() for q in P], X, y_old, y_new, act64, gate, mu_hat, xmu)
            for k in swk:
                sw[k].append(pr[k].numpy())
            sw["train_acc"].append(float(pr["train_acc"]))
            if task in SNAP:
                snaps["z_sw"].append(pr["z"].float().numpy())
            wmu0 = (P[0].detach().double() @ mu_hat).clone(); b0 = P[1].detach().double().clone()
            d_wmu, d_b, pplus_k, rho_k, m_k, zmax_k = [], [], [], [], [], []
        d_max_task, cnt_max_task, cnt = 0.0, 0, 0
        for it in range(steps):
            idx = order[it]
            xb, yb = X[idx], Y[idx]
            z = xb @ P[0].T + P[1]; a = actf(z); logits = a @ P[2].T + P[3]
            loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
            optim.zero_grad(set_to_none=True); loss.backward()
            if eps_diag:
                prev = [P[0].detach().clone(), P[1].detach().clone()]
            optim.step()
            if eps_diag:                                  # round 4: first-layer coordinates with v == 0 but m != 0, and the step size
                with torch.no_grad():
                    dmax = max(float((P[0] - prev[0]).abs().max()), float((P[1] - prev[1]).abs().max()))
                    cnt = sum(int(((optim.state[q]["exp_avg_sq"] == 0) & (optim.state[q]["exp_avg"] != 0)).sum())
                              for q in (P[0], P[1]) if q in optim.state)
                    d_max_task = max(d_max_task, dmax); cnt_max_task = max(cnt_max_task, cnt)
            if y_old is not None and (it + 1) in KS:
                with torch.no_grad():
                    d_wmu.append((P[0].double() @ mu_hat - wmu0).numpy()); d_b.append((P[1].double() - b0).numpy())
                    if it + 1 == 200:
                        st = unit_stats(P[0], P[1], X, gate)
                        pplus_k.append(st["pplus"].numpy()); rho_k.append((st["m"] / st["s"]).numpy()); m_k.append(st["m"].numpy()); zmax_k.append(st["zmax"].numpy())
                        if task in SNAP:
                            snaps["z_200"].append(st["z"].float().numpy())
        with torch.no_grad():
            st = unit_stats(P[0], P[1], X, gate)
            a = act64(st["z"]); logits = a @ P[2].double().T + P[3].double()
            ce = float((logits.logsumexp(-1) - (logits * Y.double()).sum(-1)).mean()); acc = float((logits.argmax(-1) == y_new).double().mean())
            for k in ("pplus", "gbar", "m", "s", "zmax", "zmax2", "zmin"): per_task[k].append(st[k].numpy())
            per_task["wnorm"].append(P[0].norm(dim=1).numpy().copy()); per_task["bias"].append(P[1].numpy().copy())
            per_task["disp"].append((P[0] - W_start).norm(dim=1).numpy()); per_task["dbias"].append((P[1] - b_start).numpy())
            per_task["train_acc"].append(acc); per_task["train_ce"].append(ce)
            if y_old is not None:
                d_wmu.append((P[0].double() @ mu_hat - wmu0).numpy()); d_b.append((P[1].double() - b0).numpy())
                pplus_k.append(st["pplus"].numpy()); rho_k.append((st["m"] / st["s"]).numpy()); m_k.append(st["m"].numpy()); zmax_k.append(st["zmax"].numpy())
                sw["zmax_k"].append(np.stack(zmax_k))
                sw["d_wmu"].append(np.stack(d_wmu)); sw["d_b"].append(np.stack(d_b))
                sw["pplus_k"].append(np.stack(pplus_k)); sw["rho_k"].append(np.stack(rho_k)); sw["m_k"].append(np.stack(m_k))
                if task in SNAP:
                    snaps["z_end"].append(st["z"].float().numpy())
        if eps_diag:
            diag["v0_m_nonzero_max"].append(cnt_max_task); diag["v0_m_nonzero_end"].append(cnt); diag["max_abs_dtheta"].append(d_max_task)
        y_old = y_new
        print(f"{tag} s{seed} task {task} acc {acc:.3f} ce {ce:.3f} pplus {float(st['pplus'].mean()):.4f} gbar {float(st['gbar'].mean()):.4f} "
              f"dead {float((st['pplus'] == 0).double().mean()):.2f} rho {float((st['m']/st['s']).mean()):.3f} top {float(st['zmax'].median()):.2f} ({time.monotonic()-t0:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}_s{seed}_v4.npz",
                        **{f"task_{k}": np.array(v) for k, v in per_task.items()},
                        **{f"sw_{k}": np.array(v) for k, v in sw.items()},
                        **{k: np.array(v) for k, v in snaps.items()}, snap_tasks=np.array([t for t in SNAP if t <= tasks]),
                        mu_l1_over_l2=float(mu.abs().sum() / mu.norm()), ks=np.array(KS + (steps,)), lr=lr, steps=steps, tasks=tasks,
                        alpha=alpha, dmode=dmode, act=act, flush_denormal=True, **{f"diag_{k}": np.array(v) for k, v in diag.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True); ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--dmode", default="m1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tasks", type=int, default=200); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs"); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lr1", type=float, default=None); ap.add_argument("--lr2", type=float, default=None)
    ap.add_argument("--eps1", type=float, default=None); ap.add_argument("--eps2", type=float, default=None)
    ap.add_argument("--init1", type=float, default=1.0); ap.add_argument("--init2", type=float, default=1.0)
    ap.add_argument("--freeze_bias", action="store_true"); ap.add_argument("--tag", default=None)
    ap.add_argument("--snap_every", type=int, default=0)   # sink_roots_0930 round 2 (R2_valley_long)
    ap.add_argument("--eps_diag", action="store_true")     # sink_roots_0930 round 4 (R_eps_zero_v2)
    a = ap.parse_args()
    if a.snap_every:
        SNAP = tuple(sorted(set(SNAP) | set(range(a.snap_every, a.tasks + 1, a.snap_every))))
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)   # frozen units leave Adam state stuck at float32 denormals -> 4x slower CPU; dynamics unchanged (updates ~1e-26 -> 0)
    tag = a.tag or (f"ELUA{a.alpha:g}{a.dmode}" if a.act == "ELUA" else a.act)
    for s in a.seeds:
        run(a.act, a.alpha, a.dmode, s, a.tasks, a.steps, a.lr, Path(a.out), tag,
            lr1=a.lr1, lr2=a.lr2, eps1=a.eps1, eps2=a.eps2, init1=a.init1, init2=a.init2, freeze_bias=a.freeze_bias,
            eps_diag=a.eps_diag)

#!/usr/bin/env python3
"""sink_roots_0930 round 6 (spec_sink_roots_0930_round6.md): the two-layer replay requests of the twolayer group
(derivations_round2/twolayer/final/result.md §6), from the saved switch states of round 1 (sign_states/L2_*: 2-layer
RL-MNIST 784-100-100-10, raw /255, 1,200 images, 10 labels, batch 16, Adam 1e-3 carried over, 6,000 updates per task).

The replay is replay.py's (round 1): the same batch order (env_batch_0913 advanced to the switch task), the saved
parameters and Adam state (torch Adam, float32), the switch task's actual new labels unless said otherwise.

  rr1 <state>   full arm, actual labels, 200 updates.  Before every update, on all 1,200 images in float64 at the current
                parameters: own_i = mu2~ . g2_i (mu2~ = (mean a1, 1), g2 = full-data gradient of (W2_i, b2_i)),
                up_i = sum_j W2_ij (nu_j . g1_j) (nu_j = mean_n phi1'_j(n) x~_n), gate G = mean phi1', error E = mean |eps2|
                (per-image dL/dz2), cancellation C = sum_j |<phi1'_j e_j>| / sum_j <phi1'_j sum_i |W2_ij eps2_i|>
                (e = per-image dL/da1), m1 per layer-1 unit, mu2 (for the path ledger sum_s W2(s-1) . dmu2(s)).
                The first-order ledger of the actual displacement (F1's: own = mu2~(0) . dW2~_i, up = sum_j W2_ij(0) nu_j(0) . dW1~_j)
                at tau in TAUS, and A6_B at the switch state (twolayer/final/v1_batch_content.py, ported unchanged).
                The uniform-shift predictor of the path ledger: W2(0) . [phibar(dm1(tau)) - phibar(0)], phibar_j(d) = mean_n phi(z1_jn(0) + d).
  rr2 <state>   16 uniform label draws x {nat, frozen}: 'frozen' uses, in layer 1's backward only, phi1' at the switch state's z1
                for each image (the forward is unchanged); everything is trained for 200 updates.  F1's first-order ledger at
                tau 50 / 200 (label means), dm1 per layer-1 unit and dm2 per layer-2 unit at 200.
  rr3 <state>   layer 2 reads a1 - c (c = mu2 at the switch, fixed), b2 -> b2 + W2 c (same function at the switch; Adam m, v
                unchanged); arms full and up (layer 1 only), actual labels, 200 updates; dm2 per unit at 50 / 200.
  rr4 <state>   12 uniform label draws x {vc (readout), l2 (W2, b2 and readout), full}: 6,000 updates of re-learning (Adam
                state continued); every 25 updates the shares of alive layer-2 units with S2 > 0 and S2^oh > 0 and the accuracy.
                S2_i = <K2 phi2'_i (v_i^c . p)>, S2^oh_i = <K2 phi2'_i v^c_{i,y}>, K2_n = a1_n . mu2 + 1.
Usage: python3 twolayer_probe.py <rr1|rr2|rr3|rr4> <state.npz> --out DIR
"""
import argparse, gzip, hashlib, math, re, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sink_roots_round1" / "sign"))
from common_f import activ, gate                       # noqa: E402  (the round-1 copies of the parent's definitions)

DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
N, BATCH, TSTEPS, ETA, B1, B2, EPS = 1200, 16, 6000, 1e-3, 0.9, 0.999, 1e-8
TAUS = (1, 2, 3, 5, 10, 25, 50, 100, 200)


def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def gen(tag, idx):
    h = hashlib.sha256(f"sink_roots_0930|round6|{tag}|{idx}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


AX = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255


def load(path):
    d = np.load(path, allow_pickle=True)
    L, K, act = int(d["L"]), int(d["K"]), str(d["act"])
    assert L == 2
    m = re.match(r".*_s(\d)_t(\d+)", Path(path).stem); seed, task = int(m.group(1)), int(m.group(2))
    X = torch.tensor(AX[d["subset"]])
    gbatch = stream("env_batch_0913", seed); glabel = stream("env_labels_0913", seed)
    epochs = -(-(TSTEPS * BATCH) // N)
    for t in range(1, task + 1):
        y = torch.randint(K, (N,), generator=glabel)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:TSTEPS * BATCH].reshape(TSTEPS, BATCH)
    assert (y.numpy() == d["y_new"]).all(), "the new labels are not reproduced"
    return dict(d=d, K=K, act=act, seed=seed, task=task, X=X, y=y, order=order, name=Path(path).stem)


def params(S, train):
    d = S["d"]
    P = [torch.tensor(d[f"p{i}"]).float().requires_grad_(True) for i in range(6)]
    ps = [P[i] for i in train]
    if not ps:
        return P, ps, None
    opt = torch.optim.Adam(ps, lr=ETA, betas=(B1, B2), eps=EPS)
    for i in train:
        opt.state[P[i]] = {"step": torch.tensor(float(d["step"])), "exp_avg": torch.tensor(d[f"m{i}"]).float().clone(),
                           "exp_avg_sq": torch.tensor(d[f"v{i}"]).float().clone()}
    return P, ps, opt


class FrozenGate(torch.autograd.Function):
    """forward phi(z); backward multiplies by a fixed phi' (given per row of the batch)."""
    @staticmethod
    def forward(ctx, z, act, d0):
        ctx.save_for_backward(d0)
        return activ(z, act)

    @staticmethod
    def backward(ctx, g):
        (d0,) = ctx.saved_tensors
        return g * d0, None, None


def step(P, ps, opt, X, Y, idx, act, frozen=None, center=None):
    x = X[idx]
    z1 = x @ P[0].T + P[1]
    a1 = FrozenGate.apply(z1, act, frozen[idx]) if frozen is not None else activ(z1, act)
    if center is not None:
        a1 = a1 - center
    a2 = activ(a1 @ P[2].T + P[3], act)
    lg = a2 @ P[4].T + P[5]
    loss = (lg.logsumexp(-1) - (lg * Y[idx]).sum(-1)).mean()
    gs = torch.autograd.grad(loss, ps)
    for q, g in zip(ps, gs):
        q.grad = g
    opt.step()


@torch.no_grad()
def fwd64(P, X, act, center=None):
    Xd = X.double(); W1, b1, W2, b2, W3, b3 = [q.detach().double() for q in P]
    z1 = Xd @ W1.T + b1; a1 = activ(z1, act); g1 = gate(z1, act)
    a1c = a1 - center if center is not None else a1
    z2 = a1c @ W2.T + b2; a2 = activ(z2, act); g2 = gate(z2, act)
    lg = a2 @ W3.T + b3
    return dict(Xd=Xd, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, z1=z1, a1=a1, g1=g1, z2=z2, a2=a2, g2=g2, p=lg.softmax(-1), lg=lg)


def ledger1(F0, P, P0):
    """F1's first-order ledger of the actual displacement: own_i = mu2~(0) . dW2~_i, up_i = sum_j W2(0)_ij nu_j(0) . dW1~_j."""
    mu2t = torch.cat([F0["a1"].mean(0), torch.ones(1, dtype=torch.float64)])
    Xt = torch.cat([F0["Xd"], torch.ones(N, 1, dtype=torch.float64)], 1)
    nu = (F0["g1"].T @ Xt) / N                                                       # (H1, 785)
    dW2t = torch.cat([P[2].detach().double() - P0[2], (P[3].detach().double() - P0[3])[:, None]], 1)
    dW1t = torch.cat([P[0].detach().double() - P0[0], (P[1].detach().double() - P0[1])[:, None]], 1)
    own = dW2t @ mu2t
    up = F0["W2"] @ (nu * dW1t).sum(1)
    return own.numpy(), up.numpy()


def a6b(S, F0):
    """twolayer/final/v1_batch_content.py run(), ported (A6 and A6_B at the switch state, share / own sum / up sum)."""
    d, K, act, order = S["d"], S["K"], S["act"], S["order"]
    step0 = float(d["step"]); bc2 = 1 - B2 ** step0
    W2, W3 = F0["W2"], F0["W3"]
    Xt = torch.cat([F0["Xd"], torch.ones(N, 1, dtype=torch.float64)], 1)
    a1, g1, g2, p = F0["a1"], F0["g1"], F0["g2"], F0["p"]
    Vr = W3.T; Vc = Vr - Vr.mean(1, keepdim=True)
    a1t = torch.cat([a1, torch.ones(N, 1, dtype=torch.float64)], 1); mu2t = a1t.mean(0)
    live = ((F0["z2"] > 0).sum(0) > 0)
    Ed2 = g2 * (p @ Vc.T); gb2 = Ed2.T @ a1t / N
    var2 = ((g2 ** 2 * (Vc ** 2).sum(1)[None, :] / K).T @ (a1t ** 2)) / N / BATCH + \
           ((((Ed2 ** 2).T @ (a1t ** 2)) / N - gb2 ** 2) * (N - BATCH) / (N - 1) / BATCH)
    G22 = gb2 ** 2 + var2.clamp_min(0)
    U = torch.einsum('ij,ni,ik->njk', W2, g2, Vr); Uc = U - U.mean(-1, keepdim=True)
    Ed1 = g1 * torch.einsum('njk,nk->nj', Uc, p); gb1 = Ed1.T @ Xt / N
    var1 = ((g1 ** 2 * (Uc ** 2).sum(-1) / K).T @ (Xt ** 2)) / N / BATCH + \
           ((((Ed1 ** 2).T @ (Xt ** 2)) / N - gb1 ** 2) * (N - BATCH) / (N - 1) / BATCH)
    G12 = gb1 ** 2 + var1.clamp_min(0)
    vA = [torch.tensor(d[f"v{i}"]).double() for i in range(6)]; mA = [torch.tensor(d[f"m{i}"]).double() for i in range(6)]
    V2 = torch.cat([vA[2], vA[3][:, None]], 1) / bc2; M2 = torch.cat([mA[2], mA[3][:, None]], 1)
    V1 = torch.cat([vA[0], vA[1][:, None]], 1) / bc2; M1 = torch.cat([mA[0], mA[1][:, None]], 1)
    nu = (g1.T @ Xt) / N
    def gB(idx):
        return (Ed2[idx].T @ a1t[idx] / len(idx), Ed1[idx].T @ Xt[idx] / len(idx))
    out = {}
    for tag in ("A6", "A6B"):
        d2 = torch.zeros_like(gb2); d1 = torch.zeros_like(gb1); mb2 = torch.zeros_like(gb2); mb1 = torch.zeros_like(gb1)
        res = {}
        for s in range(1, max(TAUS) + 1):
            den2 = torch.sqrt(B2 ** s * V2 + (1 - B2 ** s) * G22) + EPS
            den1 = torch.sqrt(B2 ** s * V1 + (1 - B2 ** s) * G12) + EPS
            if tag == "A6":
                n2 = (1 - B1 ** s) * gb2 + B1 ** s * M2; n1 = (1 - B1 ** s) * gb1 + B1 ** s * M1
            else:
                g2b, g1b = gB(order[s - 1])
                mb2 = B1 * mb2 + (1 - B1) * g2b; mb1 = B1 * mb1 + (1 - B1) * g1b
                n2 = mb2 + B1 ** s * M2; n1 = mb1 + B1 ** s * M1
            d2 -= ETA * n2 / den2; d1 -= ETA * n1 / den1
            if s in TAUS:
                own = (d2 @ mu2t)[live]; up = (W2 @ (nu * d1).sum(1))[live]
                res[s] = (float(up.sum() / (own.sum() + up.sum())), float(own.sum()), float(up.sum()))
        out[tag] = res
    return out, live.numpy()


def rr1(S, out):
    act, X, order = S["act"], S["X"], S["order"]
    Y = torch.nn.functional.one_hot(S["y"], S["K"]).float()
    P, ps, opt = params(S, list(range(6)))
    P0 = [q.detach().double().clone() for q in P]
    F0 = fwd64(P, X, act); z10 = F0["z1"]; m10 = z10.mean(0)
    Yd = Y.double()
    rec = {k: [] for k in ("own", "up", "G", "E", "C", "dm1", "mu2", "W2", "m2")}
    led = {}
    for s in range(201):
        F = fwd64(P, X, act)
        if s < 200:
            eps3 = (F["p"] - Yd)                                              # per image dL/dlogit (x N for the mean loss)
            eps2 = (eps3 @ F["W3"]) * F["g2"]                                 # per image dL/dz2
            e = eps2 @ F["W2"]                                                # per image dL/da1
            eps1 = e * F["g1"]
            gW2 = eps2.T @ F["a1"] / N; gb2 = eps2.mean(0); gW1 = eps1.T @ F["Xd"] / N; gb1 = eps1.mean(0)
            mu2 = F["a1"].mean(0)
            own = gW2 @ mu2 + gb2
            nug = (F["g1"] * (F["Xd"] @ gW1.T + gb1[None, :])).mean(0)       # nu_j . g1_j
            up = F["W2"] @ nug
            C = float(((F["g1"] * e).mean(0)).abs().sum() / (F["g1"] * (eps2.abs() @ F["W2"].abs())).mean(0).sum())
            for k, v in (("own", own), ("up", up)):
                rec[k].append(v.numpy())
            rec["G"].append(float(F["g1"].mean())); rec["E"].append(float(eps2.abs().mean())); rec["C"].append(C)
        rec["dm1"].append((F["z1"].mean(0) - m10).numpy()); rec["mu2"].append(F["a1"].mean(0).numpy())
        rec["W2"].append(F["W2"].numpy().astype(np.float32)); rec["m2"].append(F["z2"].mean(0).numpy())
        if s in TAUS:
            led[s] = ledger1(F0, P, P0)
        if s == 200:
            break
        step(P, ps, opt, X, Y, order[s], act)
    W2s = np.stack(rec["W2"]).astype(np.float64); mu2s = np.stack(rec["mu2"])
    up_path = np.cumsum(np.einsum("shd,sd->sh", W2s[:-1], np.diff(mu2s, axis=0)), axis=0)   # sum_s W2(s-1) . dmu2(s)
    pred = {}
    for tau in (50, 200):                                                     # uniform shift of layer-1 units by dm1(tau)
        dm = torch.tensor(rec["dm1"][tau])
        phibar = activ(z10 + dm[None, :], act).mean(0) - activ(z10, act).mean(0)
        pred[tau] = (F0["W2"] @ phibar).numpy()
    A, live = a6b(S, F0)
    dm_saved = S["d"]["dm"]                                                   # the run's own displacement (KS 50, 200, ...), per layer
    fid = max(float(np.max(np.abs(rec["m2"][50] - rec["m2"][0] - dm_saved[0, 1]))), float(np.max(np.abs(rec["m2"][200] - rec["m2"][0] - dm_saved[1, 1]))))
    np.savez_compressed(out / f"rr1_{S['name']}.npz", own=np.stack(rec["own"]), up=np.stack(rec["up"]), G=np.array(rec["G"]),
                        E=np.array(rec["E"]), C=np.array(rec["C"]), dm1=np.stack(rec["dm1"]), m2=np.stack(rec["m2"]),
                        up_path=up_path, pred50=pred[50], pred200=pred[200], live=live,
                        led_taus=np.array(sorted(led)), led_own=np.stack([led[t][0] for t in sorted(led)]),
                        led_up=np.stack([led[t][1] for t in sorted(led)]),
                        a6=np.array([A["A6"][t] for t in TAUS]), a6b=np.array([A["A6B"][t] for t in TAUS]), taus=np.array(TAUS), fid=fid)
    lv = live
    msg = " ".join(f"t{t}: real {led[t][1][lv].sum() / (led[t][0][lv].sum() + led[t][1][lv].sum()):.2f} A6B {A['A6B'][t][0]:.2f}" for t in (1, 3, 200))
    print(f"rr1 {S['name']}: replay vs saved dm2 max |diff| {fid:.1e} | up share {msg} | C(1)->C(25) {rec['C'][0]:.3f}->{rec['C'][24]:.3f}", flush=True)


def rr2(S, out, R=16):
    act, X, order, K = S["act"], S["X"], S["order"], S["K"]
    P_, _, _ = params(S, [])
    F0 = fwd64(P_, X, act)                                                  # the switch state
    d0 = F0["g1"].float()                                                   # phi1' at the switch state, per image
    res = {}
    for arm in ("nat", "frozen"):
        own_s, up_s, dm1, dm2 = {50: [], 200: []}, {50: [], 200: []}, [], []
        for r in range(R):
            y = torch.randint(K, (N,), generator=gen(f"rr2|{S['name']}", r))
            Y = torch.nn.functional.one_hot(y, K).float()
            P, ps, opt = params(S, list(range(6)))
            P0 = [q.detach().double().clone() for q in P]
            for s in range(200):
                step(P, ps, opt, X, Y, order[s], act, frozen=d0 if arm == "frozen" else None)
                if s + 1 in (50, 200):
                    o, u = ledger1(F0, P, P0); own_s[s + 1].append(o); up_s[s + 1].append(u)
            F = fwd64(P, X, act)
            dm1.append((F["z1"].mean(0) - F0["z1"].mean(0)).numpy()); dm2.append((F["z2"].mean(0) - F0["z2"].mean(0)).numpy())
        res[arm] = dict(own50=np.mean(own_s[50], 0), up50=np.mean(up_s[50], 0), own200=np.mean(own_s[200], 0), up200=np.mean(up_s[200], 0),
                        dm1=np.mean(dm1, 0), dm2=np.mean(dm2, 0), dm2_draws=np.array(dm2))
    live = ((F0["z2"] > 0).sum(0) > 0).numpy()
    np.savez_compressed(out / f"rr2_{S['name']}.npz", live=live, **{f"{a}_{k}": v for a, r_ in res.items() for k, v in r_.items()})
    sh = {a: res[a]["up200"][live].sum() / (res[a]["own200"][live].sum() + res[a]["up200"][live].sum()) for a in res}
    print(f"rr2 {S['name']}: up share tau200 nat {sh['nat']:.3f} frozen {sh['frozen']:.3f} | dm1 median nat {np.median(res['nat']['dm1']):+.3f} "
          f"frozen {np.median(res['frozen']['dm1']):+.3f} | dm2 down frozen {np.mean(res['frozen']['dm2'] < 0):.3f}", flush=True)


def rr3(S, out):
    act, X, order = S["act"], S["X"], S["order"]
    Y = torch.nn.functional.one_hot(S["y"], S["K"]).float()
    res = {}
    for arm, train in (("full", list(range(6))), ("up", [0, 1])):
        P, ps, opt = params(S, train)
        with torch.no_grad():
            c = activ(X @ P[0].T + P[1], act).double().mean(0)             # mu2 at the switch
            b2 = P[3].detach().double() + P[2].detach().double() @ c
            P[3].copy_(b2.float())
        cf = c.float()
        F0 = fwd64(P, X, act, center=c); m0 = F0["z2"].mean(0)
        rec = {}
        for s in range(200):
            step(P, ps, opt, X, Y, order[s], act, center=cf)
            if s + 1 in (50, 200):
                rec[s + 1] = (fwd64(P, X, act, center=c)["z2"].mean(0) - m0).numpy()
        res[arm] = rec
    np.savez_compressed(out / f"rr3_{S['name']}.npz", **{f"{a}_{t}": v for a, r_ in res.items() for t, v in r_.items()})
    print(f"rr3 {S['name']}: centred full dm2(200) median {np.median(res['full'][200]):+.3f} down {np.mean(res['full'][200] < 0):.2f} | "
          f"up median {np.median(res['up'][200]):+.3f} down {np.mean(res['up'][200] < 0):.2f}", flush=True)


def s2_stats(P, X, act, y, K):
    F = fwd64(P, X, act)
    mu2 = F["a1"].mean(0); K2 = F["a1"] @ mu2 + 1.0
    Vc = F["W3"] - F["W3"].mean(0, keepdim=True)                           # (K, H2): column i = v_i^c
    B = K2[:, None] * F["g2"]
    S2 = (B * (F["p"] @ Vc)).mean(0); S2oh = (B * Vc[y].double()).mean(0)
    live = (F["z2"] > 0).sum(0) > 0
    acc = float((F["lg"].argmax(-1) == y).double().mean())
    return float((S2[live] > 0).double().mean()), float((S2oh[live] > 0).double().mean()), acc, int(live.sum())


def rr4(S, out, R=12, steps=6000):
    act, X, order, K = S["act"], S["X"], S["order"], S["K"]
    arms = {"vc": [4, 5], "l2": [2, 3, 4, 5], "full": list(range(6))}
    res = {}
    for arm, train in arms.items():
        curves = []
        for r in range(R):
            g = gen(f"rr4|{S['name']}", r)
            y = torch.randint(K, (N,), generator=g); Y = torch.nn.functional.one_hot(y, K).float()
            ordr = torch.stack([torch.randperm(N, generator=g) for _ in range(-(-(steps * BATCH) // N))]).reshape(-1)[:steps * BATCH].reshape(steps, BATCH)
            P, ps, opt = params(S, train)
            cur = []
            for s in range(steps):
                step(P, ps, opt, X, Y, ordr[s], act)
                if (s + 1) % 25 == 0:
                    cur.append(s2_stats(P, X, act, y, K))
            curves.append(cur)
        res[arm] = np.array(curves)                                            # (R, steps/25, 4)
    np.savez_compressed(out / f"rr4_{S['name']}.npz", **res, every=25)
    print(f"rr4 {S['name']}: end S2>0 / S2oh>0 / acc " + " | ".join(f"{a} {v[:, -1, 0].mean():.3f}/{v[:, -1, 1].mean():.3f}/{v[:, -1, 2].mean():.2f}" for a, v in res.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rr1", "rr2", "rr3", "rr4"]); ap.add_argument("states", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for f in a.states:
        {"rr1": rr1, "rr2": rr2, "rr3": rr3, "rr4": rr4}[a.what](load(f), out)
    print(f"({time.monotonic() - t0:.0f}s)")

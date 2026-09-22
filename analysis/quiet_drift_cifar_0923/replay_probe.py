#!/usr/bin/env python3
"""quiet_drift_cifar_0923 -- replay one task from an altlabels_cifar_0923 checkpoint with probes.

Question (Issa, 0923): after the fit, W1's squared norm keeps growing ("quiet drift") about ten
times faster in the iid run than in the abab run.  Why?

This replays ONE task of 30,000 steps from ckpts/t<NN>.pt of LR_iid or LR_abab (R = 10 slots =
seeds 0-9, std), with the engine's own update written out op for op (eager; the engine's CUDA
graph is bit-identical to its eager step, check S-graph of rlcifar_mlp_battle_0918), and records:

  every step (per slot, float64):   <W1, u>, ||u||^2, ||g1||^2, <W1, g1>, <u_t, u_{t-1}>,
                                    the batch loss, the batch's largest per-sample loss,
                                    and <W, u>, ||u||^2 for W2 and W3
      where u is the Adam update the engine subtracts (W1 <- W1 - u) and g1 the minibatch gradient.
  every 100 steps (read only):      correct, CE (engine's float32 way), margin median (engine's
                                    way), n1/n2/n3  -> checked against the main run's trace;
                                    all 1200 margins (float32 logits -> float64), residuals,
                                    Adam-state statistics of W1, W1's band norms (input PCA basis),
                                    the window's summed update and summed gradient (coherence,
                                    radial part, bands), and the displacement since each slot's
                                    own hit999 + 500 (radial / fit-direction / new parts, bands).
  every 1000 steps:                 W1 itself (float32), for anything not foreseen here.

Nothing here is written back into the source run.  Labels ("branches"):
  iid net   next = its own task t+1 (reproduces the main run), same = task t's labels again,
            rev2 = task t-1's labels (seen once, two tasks ago), C = the fork C labelling
  abab net  own  = its own task t+1 (A or B; reproduces the main run), same = task t's labelling,
            C = the fork C labelling (the same C as the iid net's: one stream per (t, seed))
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import rlcifar_mlp_battle_0918 as B          # noqa: E402  the engine
from src import altlabels_cifar_0923 as AL            # noqa: E402  fork_labels / iid_labels
from src import pmnist_0905 as H                      # noqa: E402  setup
from src import pmnist_rlcifar_0907 as RC             # noqa: E402  data

SRC_ROOT = Path("/home/issan/Projects/obsidian-research-data/altlabels_cifar_0923/results/"
                "altlabels_cifar_0923")
BASIS_DIR = Path(os.environ.get(
    "QD_BASIS", "/tmp/claude-1000/-home-issan-Projects-claude/cc0afbce-b1e9-456c-95f4-43f5a640c220"
                "/scratchpad/opus_altlabels_analysis/basis_cache"))
BANDS = (("top", 0, 10), ("mid1", 10, 100), ("mid2", 100, 439), ("low", 439, 1199),
         ("mu", 1199, 1200))
BANDNAMES = tuple(n for n, _, _ in BANDS) + ("comp",)
NET_DIR = {"iid": "LR_iid", "abab": "LR_abab"}
SEEDS = list(range(10))
PROBE = 100
SNAP = 1000
HIT_NEED = 1199
HIT_PLUS = 500


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def labels_for(net: str, branch: str, t: int, st: dict) -> dict[int, torch.Tensor]:
    out = {}
    for s in SEEDS:
        if branch == "C":
            out[s] = AL.fork_labels(s, t)
        elif net == "iid":
            k = {"next": t + 1, "same": t, "rev2": t - 1}[branch]
            out[s] = AL.iid_labels(s, k)
        else:
            fixed = st["fixed_lab"][s]
            k = {"own": t + 1, "same": t}[branch]
            out[s] = fixed[B.schedule_index("abab", k)].cpu().to(torch.int64)
    return out


def load_basis(seed: int) -> torch.Tensor:
    f = BASIS_DIR / f"basis_std_s{seed}.npz"
    if not f.exists():
        raise SystemExit(f"missing basis {f} (built by altlabels_cifar_0923's ledger.py)")
    with np.load(f) as d:
        return torch.from_numpy(d["Q"])               # (3072, 1200) float64


def band_sq(Wd: torch.Tensor, Qd: torch.Tensor) -> torch.Tensor:
    """Squared norms of a (R, 100, 3072) object in each band -> (R, 6) float64."""
    C = torch.bmm(Wd, Qd)                              # (R, 100, 1200)
    out = [C[:, :, a:b].pow(2).sum((1, 2)) for _, a, b in BANDS]
    tot = Wd.pow(2).sum((1, 2))
    out.append(tot - C.pow(2).sum((1, 2)))            # comp = the rest
    return torch.stack(out, 1)


def band_dot(Ad: torch.Tensor, Bd: torch.Tensor, Qd: torch.Tensor) -> torch.Tensor:
    """<A, B> split into the bands -> (R, 6) float64 (comp = total - span part)."""
    CA, CB = torch.bmm(Ad, Qd), torch.bmm(Bd, Qd)
    out = [(CA[:, :, a:b] * CB[:, :, a:b]).sum((1, 2)) for _, a, b in BANDS]
    tot = (Ad * Bd).sum((1, 2))
    out.append(tot - (CA * CB).sum((1, 2)))
    return torch.stack(out, 1)


def qtiles(x: torch.Tensor, qs=(0.01, 0.1, 0.5, 0.9, 0.99)) -> torch.Tensor:
    """Row-wise quantiles by sorting (deterministic): x (R, n) -> (R, len(qs))."""
    xs, _ = torch.sort(x, dim=1)
    n = xs.shape[1]
    idx = torch.tensor([min(n - 1, int(round(q * (n - 1)))) for q in qs], device=x.device)
    return xs[:, idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("iid", "abab"), required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--t", type=int, default=48, help="checkpoint = end of task t")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--beta2", type=float, default=0.999,
                    help="pass 3 only: Adam's beta2 from step --beta2-from on (the engine's is "
                         "0.999; the v in hand is kept and simply decays at the new rate)")
    ap.add_argument("--beta2-from", type=int, default=1,
                    help="pass 3 only: the first step (1-based within the task) using --beta2")
    a = ap.parse_args()
    if a.net == "iid" and a.branch not in ("next", "same", "rev2", "C"):
        raise SystemExit("iid net branches: next, same, rev2, C")
    if a.net == "abab" and a.branch not in ("own", "same", "C"):
        raise SystemExit("abab net branches: own, same, C")
    torch.set_num_threads(2)
    dev = H.setup(a.device)                            # deterministic algorithms, as the engine
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = SRC_ROOT / NET_DIR[a.net] / "ckpts" / f"t{a.t:02d}.pt"
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    slots = [(q["seed"], q["cond"]) for q in st["slots"]]
    if slots != [(s, "std") for s in SEEDS]:
        raise SystemExit(f"unexpected slots {slots}")
    R = len(SEEDS)
    lab = labels_for(a.net, a.branch, a.t, st)
    # the iid net's own next labels must be the draw its restored label stream would make
    if a.net == "iid":
        for s in SEEDS:
            g = torch.Generator(device="cpu")
            g.set_state(st["g_lab"][s])
            if a.branch == "next" and not torch.equal(RC.task_labels(g), lab[s]):
                raise SystemExit(f"seed {s}: iid_labels(t+1) != the restored stream's next draw")
    cifar = RC.Cifar10()
    X = torch.stack([B.slot_inputs(cifar, s, "std", dev) for s in SEEDS])          # (R,1200,3072)
    Y = torch.stack([lab[s] for s in SEEDS]).to(dev)                                # (R, 1200)
    Q = torch.stack([load_basis(s) for s in SEEDS]).to(dev)                         # (R,3072,1200) f64

    act = B.make_act("LR")
    P = [torch.stack([st["P"][i][r] for r in range(R)]).to(dev).contiguous().requires_grad_(True)
         for i in range(6)]
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    with torch.no_grad():
        for dst, src in ((adam_m, st["m"]), (adam_v, st["v"])):
            for q, v in zip(dst, src):
                q.copy_(v.to(dev))
    tc = int(st["tc"])
    g_batch = {}
    for s in SEEDS:
        g_batch[s] = torch.Generator(device="cpu")
        g_batch[s].set_state(st["g_batch"][s])

    lr, b1, b2, eps = B.LR, 0.9, 0.999, 1e-8          # b2 is switched in the loop (pass 3)
    BATCH, N, NC = B.BATCH, B.N_IMAGES, B.N_CLASSES
    ar = torch.arange(R, device=dev)[:, None]
    static_idx = torch.zeros(R, BATCH, dtype=torch.long, device=dev)
    inv_c1 = torch.zeros((), device=dev)
    inv_c2 = torch.zeros((), device=dev)
    S = a.steps
    PS = {k: torch.zeros(S, R, dtype=torch.float64, device=dev) for k in
          ("dot_Wu", "uu", "gg", "dot_Wg", "dot_uu_prev", "loss", "loss_max",
           "n_active6", "dot_Wu2", "uu2", "dot_Wu3", "uu3")}
    u_prev = torch.zeros(R, 100, 3072, dtype=torch.float64, device=dev)
    Su = torch.zeros_like(u_prev)       # the window's summed W1 update (reset every PROBE steps)
    Sg = torch.zeros_like(u_prev)       # the window's summed W1 gradient
    Su1k = torch.zeros_like(u_prev)     # the same over 1000 steps
    win = {"uu": torch.zeros(R, dtype=torch.float64, device=dev),
           "gg": torch.zeros(R, dtype=torch.float64, device=dev),
           "uu1k": torch.zeros(R, dtype=torch.float64, device=dev)}
    W0 = P[0].detach().double().clone()                 # the task's start
    Wh = torch.zeros_like(W0)                           # each slot's own hit999 + 500 point
    h_have = [False] * R
    hit999 = [-1] * R
    Wprev = W0.clone()
    rows: dict[str, list] = {}

    def rec(name, val):
        rows.setdefault(name, []).append(val.detach().cpu().numpy() if torch.is_tensor(val)
                                         else np.asarray(val))

    @torch.no_grad()
    def probe(ts: int) -> None:
        nonlocal Wprev
        z1, a1, z2, a2, logits = B.forward(P, X, act, train=False)
        correct = (logits.argmax(-1) == Y).sum(1)
        ce32 = F.cross_entropy(logits.reshape(-1, NC), Y.reshape(-1),
                               reduction="none").view(R, N).mean(1).double()
        cor = logits.gather(2, Y[:, :, None]).squeeze(2)
        oth = logits.clone()
        oth.scatter_(2, Y[:, :, None], float("-inf"))
        marg32 = cor - oth.amax(2)
        mmed = torch.stack([marg32[r].median() for r in range(R)]).double()
        n = [(q.double() ** 2).flatten(1).sum(1) for q in (P[0], P[2], P[4])]
        # residuals in float64 from the float32 logits
        zd = logits.double()
        dz = zd - zd.gather(2, Y[:, :, None])              # z_k - z_y, 0 at k = y
        e = torch.exp(dz)
        e.scatter_(2, Y[:, :, None], 0.0)
        se = e.sum(2)
        resid = se / (1.0 + se)                            # 1 - p_y, exact in float64
        marg = -(dz.scatter(2, Y[:, :, None], float("-inf")).amax(2))
        k_eff = resid.sum(1) ** 2 / (resid.pow(2).sum(1) + 1e-300)
        rmax = resid.amax(1, keepdim=True)
        n_act = [(resid > f * rmax).sum(1) for f in (1e-1, 1e-2, 1e-4)]
        # what the float32 training gradient sees at the logits: softmax32 - onehot, per image L1
        p32 = F.softmax(logits, -1)
        oh = F.one_hot(Y, NC).to(p32.dtype)
        gl1 = (p32 - oh).abs().sum(2).double()             # (R, N)
        k_eff32 = gl1.sum(1) ** 2 / (gl1.pow(2).sum(1) + 1e-300)
        py_is_one = (p32.gather(2, Y[:, :, None]).squeeze(2) == 1.0).sum(1)
        # Adam state of W1
        vh = adam_v[0] * inv_c2 if ts > 0 else adam_v[0]
        mh = adam_m[0] * inv_c1 if ts > 0 else adam_m[0]
        sq = vh.sqrt().flatten(1)
        ratio = (mh.abs().flatten(1) / (sq + eps))
        sq_q = qtiles(sq.double())
        ratio_q = qtiles(ratio.double())
        frac_eps = torch.stack([(sq < f * eps).double().mean(1) for f in (1.0, 10.0, 100.0)], 1)
        # bands of W1, of the window's summed update / gradient, and of the displacement
        W = P[0].detach().double()
        bW = band_sq(W, Q)
        D100 = W - Wprev
        bD100 = band_sq(D100, Q)
        rad100 = band_dot(Wprev, D100, Q)
        bSu = band_sq(Su, Q)
        radSu = band_dot(Wprev, -Su, Q)
        kap_u = Su.pow(2).sum((1, 2)) / (PROBE * win["uu"] + 1e-300)
        kap_g = Sg.pow(2).sum((1, 2)) / (PROBE * win["gg"] + 1e-300)
        cos_SuW = (-(Su * Wprev).sum((1, 2))) / (Su.pow(2).sum((1, 2)).sqrt()
                                                 * Wprev.pow(2).sum((1, 2)).sqrt() + 1e-300)
        cos_SgW = ((Sg * Wprev).sum((1, 2))) / (Sg.pow(2).sum((1, 2)).sqrt()
                                                * Wprev.pow(2).sum((1, 2)).sqrt() + 1e-300)
        if ts > 0 and ts % 1000 == 0:
            kap_u1k = Su1k.pow(2).sum((1, 2)) / (1000 * win["uu1k"] + 1e-300)
            Su1k.zero_()
            win["uu1k"].zero_()
        else:
            kap_u1k = torch.full((R,), float("nan"), dtype=torch.float64, device=dev)
        # each slot's hit999 (first probe > 0 with >= 1199 right) and its own +500 reference
        cc = correct.cpu().tolist()
        for r in range(R):
            if ts > 0 and hit999[r] < 0 and cc[r] >= HIT_NEED:
                hit999[r] = ts
            if hit999[r] >= 0 and not h_have[r] and ts >= hit999[r] + HIT_PLUS:
                Wh[r] = W[r]
                h_have[r] = True
        hv = torch.tensor(h_have, device=dev)
        F_ = Wh - W0                                         # the fit's displacement (to +500)
        Dh = torch.where(hv[:, None, None], W - Wh, torch.zeros_like(W))
        rad_h = band_dot(Wh, Dh, Q)                          # <W_h, D>   per band
        new_h = band_sq(Dh, Q)                               # ||D||^2    per band
        fitdir = band_dot(F_, Dh, Q)                         # <F, D>     per band
        fit_sq = band_sq(F_, Q)
        rec("step", ts)
        rec("correct", correct)
        rec("ce32", ce32)
        rec("margin_med32", mmed)
        rec("n1", n[0]); rec("n2", n[1]); rec("n3", n[2])
        rec("margins", marg.float())                        # (R, 1200)
        rec("resid_sum", resid.sum(1)); rec("resid_max", rmax.squeeze(1))
        rec("k_eff", k_eff); rec("k_eff32", k_eff32)
        rec("n_act", torch.stack(n_act, 1))
        rec("py_is_one", py_is_one)
        rec("sqrt_v_q", sq_q); rec("ratio_q", ratio_q); rec("frac_sqrtv_lt_eps", frac_eps)
        rec("ratio_mean", ratio.double().mean(1))
        rec("band_W", bW); rec("band_D100", bD100); rec("rad100", rad100)
        rec("band_Su", bSu); rec("rad_Su", radSu)
        rec("kappa_u", kap_u); rec("kappa_g", kap_g); rec("kappa_u1k", kap_u1k)
        rec("win_uu", win["uu"].clone()); rec("win_gg", win["gg"].clone())
        rec("cos_SuW", cos_SuW); rec("cos_SgW", cos_SgW)
        rec("h_have", hv); rec("rad_h", rad_h); rec("new_h", new_h)
        rec("fitdir_h", fitdir); rec("fit_sq", fit_sq)
        Su.zero_(); Sg.zero_()
        win["uu"].zero_(); win["gg"].zero_()
        Wprev = W.clone()
        if ts % SNAP == 0:
            np.save(out / f"W1_s{ts:05d}.npy", P[0].detach().cpu().numpy())

    def step(i: int) -> None:
        xb, yb = X[ar, static_idx], Y[ar, static_idx]
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        per = F.cross_entropy(z3.reshape(-1, NC), yb.reshape(-1), reduction="none").view(R, BATCH)
        lossv = per.mean(1)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            PS["loss"][i] = lossv.double()
            PS["loss_max"][i] = per.amax(1).double()
            PS["n_active6"][i] = (per > 1e-6).sum(1).double()
            for k, (p, gr, mi, vi) in enumerate(zip(P, grads, adam_m, adam_v)):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                u = lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps)
                if k == 0:
                    pd, ud, gd = p.detach().double(), u.double(), gr.double()
                    PS["dot_Wu"][i] = (pd * ud).sum((1, 2))
                    PS["uu"][i] = ud.pow(2).sum((1, 2))
                    PS["gg"][i] = gd.pow(2).sum((1, 2))
                    PS["dot_Wg"][i] = (pd * gd).sum((1, 2))
                    PS["dot_uu_prev"][i] = (ud * u_prev).sum((1, 2))
                    u_prev.copy_(ud)
                    Su.add_(ud); Sg.add_(gd); Su1k.add_(ud)
                    win["uu"].add_(PS["uu"][i]); win["gg"].add_(PS["gg"][i])
                    win["uu1k"].add_(PS["uu"][i])
                elif k == 2:
                    PS["dot_Wu2"][i] = (p.detach().double() * u.double()).sum((1, 2))
                    PS["uu2"][i] = u.double().pow(2).sum((1, 2))
                elif k == 4:
                    PS["dot_Wu3"][i] = (p.detach().double() * u.double()).sum((1, 2))
                    PS["uu3"][i] = u.double().pow(2).sum((1, 2))
                p.sub_(u)

    t0 = time.time()
    probe(0)
    ts = 0
    spe = B.STEPS_PER_EPOCH
    n_ep = -(-S // spe)
    for e in range(n_ep):
        order = {s: torch.randperm(N, generator=g_batch[s]) for s in SEEDS}
        ORD = torch.stack([order[s] for s in SEEDS]).to(dev)
        for j in range(spe):
            if ts >= S:
                break
            static_idx.copy_(ORD[:, j * BATCH:(j + 1) * BATCH])
            tc += 1
            ts += 1
            b2 = a.beta2 if ts >= a.beta2_from else 0.999     # read by step() through the closure
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(R, BATCH, dev)
            step(ts - 1)
            if ts % PROBE == 0:
                probe(ts)
        if (e + 1) % 40 == 0:
            torch.cuda.synchronize()
            print(f"[{time.strftime('%T')}] {a.net}/{a.branch} step {ts} "
                  f"({(time.time() - t0) / ts * 1e3:.2f} ms/step)", flush=True)
    torch.cuda.synchronize()
    arrs = {k: np.stack(v) for k, v in rows.items()}
    arrs.update({f"ps_{k}": v.cpu().numpy() for k, v in PS.items()})
    arrs["hit999"] = np.asarray(hit999)
    np.savez_compressed(out / "probe.npz", **arrs)
    prov = {"experiment": "quiet_drift_cifar_0923", "net": a.net, "branch": a.branch, "t": a.t,
            "steps": S, "ckpt": str(ckpt), "ckpt_sha256": sha256_file(ckpt),
            "labels_sha256": B.labels_sha256({s: [lab[s]] for s in SEEDS}),
            "tc_start": int(st["tc"]), "tc_end": tc, "hit999": hit999, "beta2": a.beta2, "beta2_from": a.beta2_from,
            "git": B.git_state(), "seconds": time.time() - t0,
            "basis_dir": str(BASIS_DIR), "bands": BANDS,
            "torch": torch.__version__, "device": str(dev)}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print(f"done {a.net}/{a.branch}: hit999 {hit999}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

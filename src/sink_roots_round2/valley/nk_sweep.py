"""nksweep_elu_0929 (from hsweep_elu_0929 h_sweep.py, N and K exposed; N 1200 / K 10 / H 100 reproduces G1e-3): the push_lift_probe box with hidden width H as the only change (H = 100 reproduces G1e-3 of push_lift_ladder).
Extra records per task end: u = mean open units per input, share of inputs open in no unit, open mass M = mean_n sum_open (1+z),
mean (1+z) over open pairs, |v| (W2 column norms) median/mean and class-centered |v| median.  z and W2 snapshots at t50/100/200."""
import sys, argparse, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_ladder_1layer as AL
BATCH = AL.BATCH
N, K = 1200, 10
CK = (100, 200, 300, 500, 800, 1200, 2000, 4000)
MID = 500
ZSNAP = (50, 100, 200)


ACT = "elu"
FLOOR = 0.1


class EluFloor(torch.autograd.Function):
    """forward ELU (expm1, same as m1); backward max(exp(z), FLOOR) on the negative side, 1 on the positive side."""
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        return g * torch.where(z > 0, torch.ones_like(z), torch.exp(z.clamp_max(0)).clamp_min(FLOOR))


@torch.no_grad()
def stats(P, X, Y, actf, y=None):
    W1, b1, W2, b2 = [q.detach().double() for q in P]
    z = X @ W1.T + b1; a = actf(z); logits = a @ W2.T + b2; p = logits.softmax(-1)
    D = (p - Y) @ W2
    op = z > 0
    ce = float((logits.logsumexp(-1) - (logits * Y).sum(-1)).mean()); acc = float((logits.argmax(-1) == Y.argmax(-1)).double().mean())
    u = op.sum(1).double(); mass = torch.where(op, 1 + z, torch.zeros_like(z)).sum(1)
    vn = W2.norm(dim=0); Wc = W2 - W2.mean(0, keepdim=True); vc = Wc.norm(dim=0)
    fav = Wc.argmax(0); lab = Y.argmax(-1); kk = op.sum(0)
    sel = (((lab[:, None] == fav[None, :]) & op).sum(0).double() / kk.clamp_min(1).double())[kk >= 2]
    mu_u = torch.where(op, 1 + z, torch.zeros_like(z)).sum(0)[kk >= 2]
    return dict(top=z.max(0).values, body=z.median(0).values, k=op.sum(0).double(), s=z.std(0),
                A_open=(D * op).sum(0) / N, ce=ce, acc=acc, z=z,
                u=float(u.mean()), u0=float((u == 0).double().mean()), M=float(mass.mean()),
                hbar=float((1 + z)[op].mean()) if bool(op.any()) else float("nan"),
                v_med=float(vn.median()), v_mean=float(vn.mean()), vc_med=float(vc.median()), W2=W2,
                sel1=float(sel.mean()) if sel.numel() else float('nan'), mu_med=float(mu_u.median()) if mu_u.numel() else float('nan'))


def run(seed, tasks, H, lr, out, tag, N, K):
    torch.manual_seed(seed)
    ax = AL.read_idx(AL.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=AL.stream("rl_subset", seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32); del ax
    X64 = X.double()
    glabel, gbatch = AL.stream("env_labels_0913", seed), AL.stream("env_batch_0913", seed)
    P = [q.to(torch.float32).requires_grad_(True) for q in AL.initial(seed, dims=(784, H, K))]
    optim = torch.optim.Adam([{"params": [P[0], P[1]], "lr": lr}, {"params": [P[2]]}, {"params": [P[3]]}], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if ACT == "elu": actf = AL.make_act("ELUA", 1.0, "m1")
    elif ACT == "leaky": actf = AL.make_act("LR", 1.0, "m1")
    elif ACT == "elu_floor": actf = lambda z: EluFloor.apply(z)
    elif ACT in ("gelu", "silu"): actf = AL.make_act(ACT.upper(), 1.0, "m1")   # sink_roots_0930 round 2 (R1_valley_N_ladder)
    else: raise ValueError(ACT)
    keys = ("top", "body", "k", "s"); scal = ("u", "u0", "M", "hbar", "v_med", "v_mean", "vc_med", "sel1", "mu_med")
    rec = {f"{ph}_{k}": [] for ph in ("sw", "mid", "end") for k in keys}
    rec.update({"A_open": [], "ce_end": [], "acc_end": [], "ce_sw": []}); rec.update({f"end_{k}": [] for k in scal})
    zs = {}
    t0 = time.monotonic()
    for task in range(1, tasks + 1):
        y_new = torch.randint(K, (N,), generator=glabel); Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32); Y64 = Y.double()
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(-(-(4000 * BATCH) // N))]).reshape(-1)[:4000 * BATCH].reshape(4000, BATCH)
        st = stats(P, X64, Y64, actf)
        for k in keys: rec[f"sw_{k}"].append(st[k].float().numpy())
        rec["ce_sw"].append(st["ce"])
        A = []
        for it in range(4000):
            idx = order[it]; xb, yb = X[idx], Y[idx]
            z = xb @ P[0].T + P[1]; a = actf(z); logits = a @ P[2].T + P[3]
            loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            if (it + 1) in CK or (it + 1) == MID:
                st = stats(P, X64, Y64, actf)
                if (it + 1) in CK: A.append(st["A_open"].float().numpy())
                if (it + 1) == MID:
                    for k in keys: rec[f"mid_{k}"].append(st[k].float().numpy())
        for k in keys: rec[f"end_{k}"].append(st[k].float().numpy())
        for k in scal: rec[f"end_{k}"].append(st[k])
        rec["A_open"].append(np.stack(A)); rec["ce_end"].append(st["ce"]); rec["acc_end"].append(st["acc"])
        if task in ZSNAP: zs[f"t{task}_z_end"] = st["z"].float().numpy(); zs[f"t{task}_W2"] = st["W2"].float().numpy()
        if task % 20 == 0 or task <= 2:
            al = rec["sw_k"][-1] > 0
            print(f"{tag} s{seed} t{task}: alive {al.mean():.2f} k_end(alive) {np.median(rec['end_k'][-1][al]) if al.any() else 0:.0f} u {st['u']:.2f} M {st['M']:.1f} hbar {st['hbar']:.1f} |v| {st['v_med']:.2f} ce {st['ce']:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}_s{seed}.npz", ck=np.array(CK), mid=MID, lr=lr, H=H, N=N, K=K, tasks=tasks, **{k: np.array(v) for k, v in rec.items()}, **zs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--H", type=int, default=100); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--N", type=int, default=1200); ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--act", default="elu")
    ap.add_argument("--tag", required=True); ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_flush_denormal(True)
    N, K = a.N, a.K
    ACT = a.act
    run(a.seed, a.tasks, a.H, a.lr, Path(a.out), a.tag, a.N, a.K)

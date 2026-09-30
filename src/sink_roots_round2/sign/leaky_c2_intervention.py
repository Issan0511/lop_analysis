"""sink_roots_0930 round 2, leaky_C2_intervention (spec_sink_roots_0930_round2.md).
Does the upward push of leaky units come from C2 = Cov_n(K, phi)/Kbar * sum K phi'  (antiparallel x leak x closed mass)?
Intervention on the leak only: from a saved leaky (a = 0.1) switch state, the first layer is fixed (z fixed); the output of
closed inputs is replaced by a' z (gate a'), a' in {0.02, 0.1, 0.3}; the readout (W2, b2) alone is re-learned with Adam
(1e-3, batch 16, saved Adam state of the readout continued, as in df_test.py) for 4,000 updates on a label set.
At the re-learned readout (definitions of leaky_c2.py / filter_real.py):
  S_i  = sum_n K_n phi'_ni (p_n . v^c_i)            expected push under uniform new labels; up = S_i < 0
  C1_i = sum_n K_n phi'_ni (phi_ni - phiK_i),  C2_i = (phiK_i - phibar_i) sum_n K_n phi'_ni,  Mc = C1 + C2
  Cov_n(K, phi) per unit, pi+ = K-weighted open share, K_n = x_n . mean(x) + 1
Label sets: the state's own y_old ("actual") and 16 independent uniform sets (the re-learned task's labels, as the
R = 12 draws of df_test.py).  The batch order of a set is shared by the three a' (paired comparison).
Also: the same quantities at the saved readout without re-learning ("norefit").
Usage: python3 leaky_c2_intervention.py <state.npz> [<state.npz> ...] --out DIR
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_f import load_state
torch.set_num_threads(1)
N, BATCH, STEPS, NSETS = 1200, 16, 4000, 16
APRIME = (0.02, 0.1, 0.3)


def gen(name, idx):
    h = hashlib.sha256(f"sink_roots_0930|leaky_c2|{name}|{idx}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def measures(z, a, g, Kn, V, c):
    """z, a (activation), g (gate): [N, H]; V: [K, H] readout, c: [K]."""
    lg = a @ V.T + c; p = lg.softmax(-1)
    Vc = V.T - V.T.mean(1, keepdim=True)                     # [H, K]
    Bm = Kn[:, None] * g
    S = (Bm * (p @ Vc.T)).sum(0)
    aK = (Kn[:, None] * a).sum(0) / Kn.sum(); ab = a.mean(0)
    C1 = (Bm * (a - aK)).sum(0); C2 = (aK - ab) * Bm.sum(0)
    covKa = (((Kn - Kn.mean())[:, None]) * (a - ab)).mean(0)
    pip = (Kn[:, None] * (z > 0)).sum(0) / Kn.sum()
    alive = (z > 0).sum(0) > 0
    return {k: v.numpy() for k, v in dict(S=S, C1=C1, C2=C2, Mc=C1 + C2, covKa=covKa, pip=pip, alive=alive).items()}


def summary(m):
    al = m["alive"]; up = al & (m["S"] < 0); upm = al & (m["Mc"] < 0)
    return dict(alive=int(al.sum()), up=float(up.sum() / max(al.sum(), 1)), up_Mc=float(upm.sum() / max(al.sum(), 1)),
                up_cov_neg=float(np.mean(m["covKa"][up] < 0)) if up.any() else float("nan"),
                up_pip_med=float(np.median(m["pip"][up])) if up.any() else float("nan"),
                down_pip_med=float(np.median(m["pip"][al & ~up])) if (al & ~up).any() else float("nan"),
                upMc_cov_neg=float(np.mean(m["covKa"][upm] < 0)) if upm.any() else float("nan"),
                S_up_Mc_agree=float(np.mean((m["S"][al] < 0) == (m["Mc"][al] < 0))),
                C2_over_C1_med=float(np.median(np.abs(m["C2"][al]) / (np.abs(m["C1"][al]) + 1e-300))))


def run_state(path, out):
    st = load_state(path); name = Path(path).stem
    assert st["act"] == "LR" and st["L"] == 1, (st["act"], st["L"])
    X, (W1, b1, V0, c0), K = st["X"], st["P"], st["K"]
    z = X @ W1.T + b1; Kn = X @ X.mean(0) + 1.0
    sets = [("actual", st["y_old"])] + [(f"r{j:02d}", torch.randint(K, (N,), generator=gen(name, 1000 + j))) for j in range(NSETS)]
    rec = {"state": name, "rows": []}; arrs = {}
    for ap in APRIME:
        a = torch.where(z > 0, z, ap * z); g = torch.where(z > 0, torch.ones_like(z), torch.full_like(z, ap))
        m = measures(z, a, g, Kn, V0, c0)
        rec["rows"].append(dict(aprime=ap, set="norefit", **summary(m)))
        for k, v in m.items(): arrs.setdefault(f"norefit_{k}", []).append(v)
        for si, (sname, y) in enumerate(sets):
            Y = torch.nn.functional.one_hot(y, K).double()
            V = V0.clone().requires_grad_(True); c = c0.clone().requires_grad_(True)
            opt = torch.optim.Adam([V, c], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
            for q, j in ((V, 2), (c, 3)):
                opt.state[q] = {"step": torch.tensor(st["step"]), "exp_avg": st["mA"][j].clone(), "exp_avg_sq": st["vA"][j].clone()}
            gb = gen(name, si)
            order = torch.stack([torch.randperm(N, generator=gb) for _ in range(-(-(STEPS * BATCH) // N))]).reshape(-1)[:STEPS * BATCH].reshape(STEPS, BATCH)
            for it in range(STEPS):
                idx = order[it]
                lg = a[idx] @ V.T + c
                loss = (lg.logsumexp(-1) - (lg * Y[idx]).sum(-1)).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                m = measures(z, a, g, Kn, V.detach(), c.detach())
                acc = float(((a @ V.T + c).argmax(-1) == y).double().mean())
            rec["rows"].append(dict(aprime=ap, set=sname, acc=acc, **summary(m)))
            for k, v in m.items(): arrs.setdefault(f"refit_{k}", []).append(v)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"c2int_{name}.npz", aprime=np.array(APRIME), sets=np.array([s for s, _ in sets]),
                        **{k: np.array(v).reshape((len(APRIME), -1) + np.array(v).shape[1:]) for k, v in arrs.items()})
    (out / f"c2int_{name}.json").write_text(json.dumps(rec, indent=1))
    for ap in APRIME:
        rs = [r for r in rec["rows"] if r["aprime"] == ap]
        rr = [r for r in rs if r["set"].startswith("r")]; ra = [r for r in rs if r["set"] == "actual"][0]; rn = [r for r in rs if r["set"] == "norefit"][0]
        print(f"{name} a'={ap:<5g} | norefit up {rn['up']:.3f} | actual up {ra['up']:.3f} (Mc {ra['up_Mc']:.3f}) cov<0 {ra['up_cov_neg']:.2f} pi+ up {ra['up_pip_med']:.3f} acc {ra['acc']:.2f}"
              f" | 16 sets up {np.mean([r['up'] for r in rr]):.3f} [{min(r['up'] for r in rr):.3f}, {max(r['up'] for r in rr):.3f}]"
              f" cov<0 {np.nanmean([r['up_cov_neg'] for r in rr]):.2f} pi+ up {np.nanmedian([r['up_pip_med'] for r in rr]):.3f}"
              f" |C2|/|C1| {np.median([r['C2_over_C1_med'] for r in rr]):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    for s in a.states:
        run_state(s, Path(a.out))
    print(f"({time.monotonic() - t0:.0f}s)")

#!/usr/bin/env python3
"""head0_mlp_1108: MLP の切替で head だけ 0 に戻す対照（spec specs/spec_head0_mlp_1108.md）。

箱と乱数列は obsidian-research-data/push_direction_proof_0930/gen_states.py
（sha256 4571091d4c02ef6a3d74457a62ad128f2ae177525d8a48d47f2ed70b0fec3f30）と同じ:
784-100-10、MNIST raw/255・subset 1,200、乱数ラベル K = 10、Adam 1e-3 (0.9, 0.999, 1e-8)・batch 16・CPU 1 スレッド・float32。
helper は src/reload_dial_1020.py から写した（S-replay・S-switch の完全一致が通ったもの）。

本線 S は engine と同じ θ・Adam を 30 課題回し、課題 t ≥ 2 の頭で snap した複製から枝の腕を回す。
腕: H0（head と head の Adam を 0）・H0b（同、別のミニバッチ順）・H0k（head の重みだけ 0、Adam は全部残す）・
    H0A（head を 0、Adam を全部新品）・Sb（そのまま、別の順）・fa/fb（新品の初期値と新品の Adam）。
"""
import argparse, gzip, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
SAVED = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/states")
WORKTREE = Path(__file__).resolve().parent.parent
N, BATCH, H, K = 1200, 16, 100, 10
LR, BETAS, EPS = 1e-3, (0.9, 0.999), 1e-8
ENGINE_STEPS = 4000
EVAL_EVERY = 10
MID_AT = 200
SQRT2 = math.sqrt(2.0)


# ---- helpers copied verbatim from src/reload_dial_1020.py ----
def stream(role, seed):                       # gen_states.py と同一（engine の乱数列）
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def activ(z, act):
    if act == "LR":   return torch.where(z > 0, z, z * 0.1)
    if act == "ELU":  return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    if act == "R":    return torch.relu(z)
    if act == "GELU": return z * 0.5 * (1.0 + torch.erf(z / SQRT2))
    if act == "SILU": return z * torch.sigmoid(z)
    raise ValueError(act)


def initial(seed, dims):
    g = stream("init", seed); p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p


def load_X(seed):
    torch.manual_seed(seed)                                   # gen_states.py と同じ順で呼ぶ
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", seed))[:N].numpy()
    return torch.tensor(ax[subset], dtype=torch.float32), subset


def order_for(g, steps):
    epochs = -(-(steps * BATCH) // N)
    return torch.stack([torch.randperm(N, generator=g) for _ in range(epochs)]).reshape(-1)[:steps * BATCH].reshape(steps, BATCH)


def step(P, optim, X, Y, idx, act):
    xb, yb = X[idx], Y[idx]
    z = xb @ P[0].T + P[1]; a = activ(z, act); logits = a @ P[2].T + P[3]
    loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
    optim.zero_grad(set_to_none=True); loss.backward(); optim.step()


def accuracy(P, X, y, act):
    with torch.no_grad():
        lg = activ(X @ P[0].T + P[1], act) @ P[2].T + P[3]
        return float((lg.argmax(-1) == y).float().mean())


def snap(P, optim, it, acc):
    st = optim.state_dict()["state"]
    d = {f"p{q}": P[q].detach().numpy().copy() for q in range(4)}
    for q in range(4):
        s = st.get(q)
        d[f"m{q}"] = (s["exp_avg"].numpy().copy() if s else np.zeros_like(d[f"p{q}"]))
        d[f"v{q}"] = (s["exp_avg_sq"].numpy().copy() if s else np.zeros_like(d[f"p{q}"]))
        d[f"step{q}"] = np.array(float(s["step"]) if s else 0.0)
    d["update"] = np.array(it); d["acc"] = np.array(acc)
    return d


def restore(ck):
    P = [torch.tensor(ck[f"p{q}"], dtype=torch.float32).requires_grad_(True) for q in range(4)]
    optim = torch.optim.Adam(P, lr=LR, betas=BETAS, eps=EPS)
    if float(ck["step0"]) > 0:
        for q in range(4):
            optim.state[P[q]] = {"step": torch.tensor(float(ck[f"step{q}"])),
                                 "exp_avg": torch.tensor(ck[f"m{q}"]), "exp_avg_sq": torch.tensor(ck[f"v{q}"])}
    return P, optim
# ---- end of copied helpers ----


def h0stream(role, seed):                     # この走だけの乱数列（engine の列と衝突しない）
    h = hashlib.sha256(f"head0_mlp_1108|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def initial_from(g, dims=(784, H, K)):        # initial() と同じ分布・同じ引く順、乱数列だけ g
    p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p


def new_params(p):
    return [q.to(dtype=torch.float32).requires_grad_(True) for q in p]


def new_adam(P):
    return torch.optim.Adam(P, lr=LR, betas=BETAS, eps=EPS)


def maxabs(P, ref):                           # ref: 4 個の numpy 配列
    return max(float(np.abs(P[q].detach().numpy() - ref[q]).max()) for q in range(4))


class Ctx:
    def __init__(self, X, act):
        self.X, self.act = X, act
        self.Xd = X.double()
        self.xbar = self.Xd.mean(0)


def evaluate(P, ctx, y):
    with torch.no_grad():
        lg = activ(ctx.X @ P[0].T + P[1], ctx.act) @ P[2].T + P[3]
        acc = (lg.argmax(-1) == y).float().mean()
        ce = (lg.logsumexp(-1) - lg.gather(1, y[:, None]).squeeze(1)).mean()
    return acc, ce


def unit_stats(P, ctx):
    with torch.no_grad():
        W1, b1 = P[0].detach().double(), P[1].detach().double()
        m = W1 @ ctx.xbar + b1
        Z = ctx.Xd @ W1.T + b1
        pplus = (Z > 0).double().mean(0)
        sigma = Z.std(0, unbiased=False)
        return np.stack([m.numpy(), pplus.numpy(), sigma.numpy()]), float(P[2].detach().double().norm())


def run_arm(P, optim, ctx, y, Y, order, steps, probe_body_grad=False, mid_at=None):
    """engine の step() と同じ演算で steps 更新回し、記録を返す。probe_body_grad は最初の 1 更新の体の勾配を読む。"""
    act, X = ctx.act, ctx.X
    online = []
    ev = torch.zeros(steps // EVAL_EVERY + 1, 2)
    u0, h0 = unit_stats(P, ctx)
    a, c = evaluate(P, ctx, y); ev[0, 0], ev[0, 1] = a, c
    body_grad, mid = None, None
    for it in range(steps):
        idx = order[it]
        xb, yb = X[idx], Y[idx]
        z = xb @ P[0].T + P[1]; a = activ(z, act); logits = a @ P[2].T + P[3]
        loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
        optim.zero_grad(set_to_none=True); loss.backward()
        if probe_body_grad and it == 0:
            body_grad = float(P[0].grad.abs().sum() + P[1].grad.abs().sum())
        optim.step()
        online.append(loss.detach())
        if mid_at is not None and it + 1 == mid_at:
            mid = [q.detach().numpy().copy() for q in P]
        if (it + 1) % EVAL_EVERY == 0:
            a, c = evaluate(P, ctx, y); k = (it + 1) // EVAL_EVERY
            ev[k, 0], ev[k, 1] = a, c
    u1, h1 = unit_stats(P, ctx)
    rec = dict(online=torch.stack(online).numpy().astype(np.float32), eval=ev.numpy().astype(np.float32),
               unit=np.stack([u0, u1]).astype(np.float64), head=np.array([h0, h1], dtype=np.float64))
    return rec, body_grad, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True, choices=["LR", "ELU", "R", "GELU", "SILU"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results/head0_mlp_1108/raw")
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--steps", type=int, default=ENGINE_STEPS)
    ap.add_argument("--rep-from", type=int, default=21)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.tasks, a.steps, a.rep_from, a.out = 3, 300, 3, "results/_smoke/head0_mlp_1108/raw"
    out = Path(a.out)
    if not out.is_absolute(): out = WORKTREE / out
    act, seed, tasks, steps, rep_from = a.act, a.seed, a.tasks, a.steps, a.rep_from
    assert steps % EVAL_EVERY == 0
    t_start = time.monotonic()

    X, subset = load_X(seed)
    saved = np.load(SAVED / f"{act}_K10_s{seed}.npz")
    assert np.array_equal(saved["subset"], subset), "subset が保存済みと違う"
    n_saved = saved["y"].shape[0]
    ctx = Ctx(X, act)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P = new_params(initial(seed, (784, H, K)))
    optim = new_adam(P)
    init_ref = [saved[f"init{q}"] for q in range(4)]

    full_engine = (steps == ENGINE_STEPS)
    check = dict(init_equal=maxabs(P, init_ref) == 0.0,
                 end_check_applies=full_engine,
                 end_check_note=("compared with saved end*[t]" if full_engine else
                                 f"steps={steps} != {ENGINE_STEPS}: end-of-task states are not comparable; end check skipped (NaN)"),
                 mid_check_note=("mid*[t] compared at update 200 of every task" if full_engine else
                                 "steps != 4000: only task 1's mid is comparable (later tasks start from a different state); others recorded but not asserted"),
                 order_note=f"engine order drawn with order_for(gbatch, max(steps, {ENGINE_STEPS})) and truncated to steps, so gbatch stays aligned with the engine")
    check_end = np.full(tasks, np.nan); check_mid = np.full(tasks, np.nan)
    res = {}; task_meta = []
    fresh_init = {}

    for t in range(1, tasks + 1):
        t_task = time.monotonic()
        y = torch.randint(K, (N,), generator=glabel)
        order = order_for(gbatch, max(steps, ENGINE_STEPS))[:steps]
        if t <= n_saved:
            assert np.array_equal(y.numpy(), saved["y"][t - 1]), f"課題 {t} のラベルが保存済みと違う"
        Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
        rep = t >= rep_from
        order_b = order_for(h0stream(f"batch_b|{t}", seed), steps) if rep else None
        arms = []

        def put(arm, rec):
            for k, v in rec.items(): res[f"{arm}_t{t:02d}_{k}"] = v
            arms.append(arm)

        if t >= 2:
            ck = snap(P, optim, (t - 1) * steps, float("nan"))
            # H0: head = 0、head の Adam 状態を捨てる（最初の step で step=0・m=v=0 から作り直される）
            def make_h0(keep_head_adam=False, fresh_adam=False):
                if fresh_adam:
                    ck2 = dict(ck); ck2.update({f"step{q}": np.array(0.0) for q in range(4)})
                    Pb, ob = restore(ck2)
                else:
                    Pb, ob = restore(ck)
                with torch.no_grad():
                    Pb[2].data.zero_(); Pb[3].data.zero_()
                if not keep_head_adam and not fresh_adam:
                    ob.state.pop(Pb[2], None); ob.state.pop(Pb[3], None)
                return Pb, ob
            Pb, ob = make_h0()
            probe = "h0_first_body_grad" not in check
            rec, bg, _ = run_arm(Pb, ob, ctx, y, Y, order, steps, probe_body_grad=probe)
            if probe: check["h0_first_body_grad"] = bg; check["h0_first_body_grad_task"] = t
            put("H0", rec)
            if rep:
                Pb, ob = make_h0()
                put("H0b", run_arm(Pb, ob, ctx, y, Y, order_b, steps)[0])
                Pb, ob = make_h0(keep_head_adam=True)
                put("H0k", run_arm(Pb, ob, ctx, y, Y, order, steps)[0])
            Pb, ob = make_h0(fresh_adam=True)
            put("H0A", run_arm(Pb, ob, ctx, y, Y, order, steps)[0])
            if rep:
                Pb, ob = restore(ck)
                put("Sb", run_arm(Pb, ob, ctx, y, Y, order_b, steps)[0])
            if t == 2 and steps >= MID_AT:   # C2 変異対照: Adam 状態を捨てて同じ 200 更新
                ck_fresh = dict(ck); ck_fresh.update({f"step{q}": np.array(0.0) for q in range(4)})
                Pm, om = restore(ck_fresh)
                for it in range(MID_AT): step(Pm, om, X, Y, order[it], act)
                check["mid1_mutant_maxabs"] = maxabs(Pm, [saved[f"mid{q}"][1] for q in range(4)])
            del ck

        for tag in (["a", "b"] if rep else ["a"]):
            p0 = initial_from(h0stream(f"fresh_init|{tag}|{t}", seed))
            Pf = new_params(p0)
            fresh_init[(tag, t)] = [q.detach().numpy().copy() for q in Pf]
            put(f"f{tag}", run_arm(Pf, new_adam(Pf), ctx, y, Y, order, steps)[0])
        if t == 1:
            fi = fresh_init[("a", 1)]
            check["fresh_a_t01_vs_engine_init_maxabs"] = max(float(np.abs(fi[q] - init_ref[q]).max()) for q in range(4))
            check["fresh_a_t01_vs_engine_init_min_over_params"] = min(float(np.abs(fi[q] - init_ref[q]).max()) for q in range(4))
        if rep and "fresh_a_vs_b_maxabs" not in check:
            fa_, fb_ = fresh_init[("a", t)], fresh_init[("b", t)]
            check["fresh_a_vs_b_maxabs"] = max(float(np.abs(fa_[q] - fb_[q]).max()) for q in range(4))
            check["fresh_a_vs_b_min_over_params"] = min(float(np.abs(fa_[q] - fb_[q]).max()) for q in range(4))
            check["fresh_a_vs_b_task"] = t
        for k in [k for k in fresh_init if k[1] == t and k != ("a", 1)]: fresh_init.pop(k)

        # 本線 S（engine の軌道を進める）
        rec, _, mid = run_arm(P, optim, ctx, y, Y, order, steps, mid_at=MID_AT if steps >= MID_AT else None)
        put("S", rec)
        if t <= n_saved:
            if mid is not None:
                check_mid[t - 1] = max(float(np.abs(mid[q] - saved[f"mid{q}"][t - 1]).max()) for q in range(4))
            if full_engine:
                check_end[t - 1] = maxabs(P, [saved[f"end{q}"][t - 1] for q in range(4)])
        el = time.monotonic() - t_task
        accS = float(rec["eval"][-1, 0])
        task_meta.append(dict(task=t, arms=arms, elapsed_s=round(el, 2), S_acc_end=accS,
                              saved_acc=float(saved["acc"][t - 1]) if t <= n_saved else None))
        print(f"{act} s{seed} task {t:2d} S acc {accS:.3f} end_maxabs {check_end[t-1]} mid_maxabs {check_mid[t-1]} "
              f"arms {','.join(arms)} ({el:.1f}s, total {time.monotonic()-t_start:.0f}s)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    res["y"] = saved["y"][:tasks].copy() if tasks <= n_saved else np.array([])
    res["check_end_maxabs"] = check_end; res["check_mid_maxabs"] = check_mid
    np.savez_compressed(out / f"{act}_s{seed}.npz", **res)
    try:
        head = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        head = f"error: {e}"
    check["check_end_maxabs"] = [None if np.isnan(v) else v for v in check_end.tolist()]
    check["check_mid_maxabs"] = [None if np.isnan(v) else v for v in check_mid.tolist()]
    meta = dict(act=act, seed=seed, tasks=tasks, steps=steps, rep_from=rep_from, smoke=a.smoke,
                torch=torch.__version__, git_head=head, engine_sha256="4571091d4c02ef6a3d74457a62ad128f2ae177525d8a48d47f2ed70b0fec3f30",
                elapsed_s=round(time.monotonic() - t_start, 1), per_task=task_meta, check=check)
    (out / f"{act}_s{seed}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(dict(act=act, seed=seed, check={k: v for k, v in check.items() if not k.endswith("note")})), flush=True)

    # 検査（出力を書いた後で走を殺す）
    assert check["init_equal"], "engine の初期値と一致しない"
    if full_engine:
        assert np.all(check_end[:min(tasks, n_saved)] == 0.0), f"C1 end 不一致 {check_end}"
        assert np.all(check_mid[:min(tasks, n_saved)] == 0.0), f"C1 mid 不一致 {check_mid}"
    elif steps >= MID_AT:
        assert check_mid[0] == 0.0, f"C1 mid（課題 1）不一致 {check_mid[0]}"
    if "mid1_mutant_maxabs" in check:
        assert check["mid1_mutant_maxabs"] > 0, "C2 変異対照が一致してしまった"
    assert check["fresh_a_t01_vs_engine_init_min_over_params"] > 0, "C3 fa が engine の初期値と一致"
    if "fresh_a_vs_b_min_over_params" in check:
        assert check["fresh_a_vs_b_min_over_params"] > 0, "C3 fa と fb が一致"
    if "h0_first_body_grad" in check:
        assert check["h0_first_body_grad"] == 0.0, f"C4 体の勾配が 0 でない {check['h0_first_body_grad']}"
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()

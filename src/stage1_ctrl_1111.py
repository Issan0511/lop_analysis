#!/usr/bin/env python3
"""stage1_ctrl_1111: 段 ①（学んだ読み出しが切替の押しを作る）の対照（spec specs/spec_stage1_ctrl_1111.md）。

箱と乱数列は obsidian-research-data/push_direction_proof_0930/gen_states.py
（sha256 4571091d4c02ef6a3d74457a62ad128f2ae177525d8a48d47f2ed70b0fec3f30）と同じ:
784-100-10、MNIST raw/255・subset 1,200、乱数ラベル K = 10、batch 16・4,000 更新/課題・30 課題、CPU 1 スレッド・float32。
helper は src/head0_mlp_1108.py（= src/reload_dial_1020.py から写したもの）から写した。

mode
  branch  本線 S（最適化器 adam / sgd / adaml）を 30 課題回し、課題 t >= 2 の頭で複製した状態から
          腕を 1 課題ずつ回す（課題 t のラベルとミニバッチの順は本線と同じ）。fa は新品の網。
  persist 腕の介入を、全ての切替（t >= 2）で施しながら 30 課題を回す軌道。
  dial    reload_dial_1020 の課題 1 の途中の checkpoint から、読み出しを替えて新ラベルへ切り替え 200 更新。

腕（branch。介入は課題 t の頭、更新 0 の前に一度だけ施す）
  S    本線そのもの
  P0   全ユニットで同じクラスの並べ替え π（読み出しの行・出力の bias・それらの Adam の状態を同じ π で並べ替える。
       ラベルの付け替えと厳密に同じ網で、S と統計的に同じ。帰無の腕）
  P0m  P0 と同じ並べ替えを読み出しにだけ施し、出力の bias は「標本で平均した logit を保つ」規則で直す（P の bias の規則の帰無）
  P    ユニットごとに独立なクラスの並べ替え（各ユニットの読み出しの値の集合とノルムは同じ・前のラベルとの対応だけ消える）。
       読み出しの Adam の状態も同じ並べ替え。bias は標本で平均した logit を保つ: b <- b + (V - V')ā
  Q    各ユニットの読み出しの中心化した成分を、その seed の初期化の時点のノルムへ縮める（向きは保つ）。bias は P と同じ規則。Adam は保つ
  R    出力層（読み出しと bias）を初期化と同じ分布から引き直し、出力層の Adam の状態を捨てる
  H0   出力層を 0 にし、出力層の Adam の状態を捨てる（head0_mlp_1108 の H0 と同じ）
  H0A  出力層を 0 にし、全ての Adam の状態を捨てる（head0_mlp_1108 の H0A と同じ）
  A0   隠れ層の Adam の状態だけを捨てる（出力層と出力層の Adam は保つ）
  F200 最初の 200 更新は隠れ層の勾配を捨てる（隠れ層の重み・bias・Adam の状態は動かない）。その後は普通に学ぶ
  fa   新品の網（専用の乱数列の初期値・新品の最適化器）
"""
import argparse, gzip, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
SAVED = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/states")
DIAL = Path("/home/issan/Projects/obsidian-research-data/reload_dial_1020/raw/ckpt")
WORKTREE = Path(__file__).resolve().parent.parent
N, BATCH, H, K = 1200, 16, 100, 10
BETAS, EPS = (0.9, 0.999), 1e-8
ENGINE_STEPS = 4000
BIN = 100                      # online の集計の幅（更新）
EVAL_EVERY = 100
FIRST = 200                    # online の損失を 1 更新ごとに残す長さ・F200 の凍結の長さ・沈みを測る点
SQRT2 = math.sqrt(2.0)
BRANCH_ARMS = ("P0", "P0m", "P", "Q", "R", "H0", "H0A", "A0", "F200")
PERSIST_ARMS = ("P", "Q", "R", "H0", "A0", "F200")
ENGINE_SHA = "4571091d4c02ef6a3d74457a62ad128f2ae177525d8a48d47f2ed70b0fec3f30"


# ---- helpers copied verbatim from src/head0_mlp_1108.py (= src/reload_dial_1020.py) ----
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
# ---- end of copied helpers ----


def dphi64(z, act):                           # 活性化の微分（float64・解析形）
    if act == "LR":   return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
    if act == "R":    return (z > 0).double()
    if act == "ELU":  return torch.where(z > 0, torch.ones_like(z), torch.exp(z.clamp_max(0)))
    if act == "GELU": return 0.5 * (1.0 + torch.erf(z / SQRT2)) + z * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    if act == "SILU":
        s = torch.sigmoid(z); return s * (1 + z * (1 - s))
    raise ValueError(act)


def own(role, seed):                          # この走だけの乱数列（engine の列と衝突しない）
    h = hashlib.sha256(f"stage1_ctrl_1111|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def initial_from(g, dims=(784, H, K)):        # initial() と同じ分布・同じ引く順、乱数列だけ g
    p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p


# ---------------- optimizers ----------------
class AdamL(torch.optim.Optimizer):
    """Lyle ら 2023 §4.1 の設定: β2 = 0.9・平方根の内側の ε = 1e-3（外側は 1e-8）。"""
    def __init__(self, params, lr, betas=(0.9, 0.9), eps_in=1e-3, eps_out=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps_in=eps_in, eps_out=eps_out))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            b1, b2 = grp["betas"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = torch.tensor(0.0); st["exp_avg"] = torch.zeros_like(p); st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1; t = float(st["step"])
                st["exp_avg"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mhat = st["exp_avg"] / (1 - b1 ** t); vhat = st["exp_avg_sq"] / (1 - b2 ** t)
                p.add_(-grp["lr"] * mhat / ((vhat + grp["eps_in"]).sqrt() + grp["eps_out"]))


def make_opt(kind, P, lr):
    if kind == "adam":  return torch.optim.Adam(P, lr=lr, betas=BETAS, eps=EPS)   # engine と同じ
    if kind == "sgd":   return torch.optim.SGD(P, lr=lr)                         # engine の --opt sgd と同じ
    if kind == "adaml": return AdamL(P, lr=lr)
    raise ValueError(kind)


def new_params(p):
    return [q.detach().clone().to(dtype=torch.float32).requires_grad_(True) for q in p]


def snapshot(P, opt):
    sp = [q.detach().clone() for q in P]
    so = []
    for q in P:
        s = opt.state.get(q)
        so.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()} if s else None)
    return sp, so


def restore(snap, kind, lr):
    sp, so = snap
    P = [q.clone().requires_grad_(True) for q in sp]
    opt = make_opt(kind, P, lr)
    for q, s in zip(P, so):
        if s:
            opt.state[q] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
    return P, opt


def maxabs(P, ref):
    return max(float(np.abs(P[q].detach().numpy() - ref[q]).max()) for q in range(4))


# ---------------- measurements ----------------
UPTS = (0, 1, 10, 50, 200, 1000, 4000)


class Ctx:
    def __init__(self, X, act):
        self.X, self.act = X, act
        self.Xd = X.double()
        self.xbar = self.Xd.mean(0)
        self.Kn = self.Xd @ self.xbar + 1.0


def unit_stats(P, ctx):
    """前活性の平均 m・標準偏差・開口率 p+・|φ'| の平均・|φ'| の最大（1,200 枚・float64）。"""
    with torch.no_grad():
        W1, b1 = P[0].detach().double(), P[1].detach().double()
        Z = ctx.Xd @ W1.T + b1
        d = dphi64(Z, ctx.act).abs()
        return np.stack([Z.mean(0).numpy(), Z.std(0, unbiased=False).numpy(), (Z > 0).double().mean(0).numpy(),
                         d.mean(0).numpy(), d.max(0).values.numpy()]).astype(np.float32)


def expected_push(P, ctx):
    """式 2 の切替の期待の勾配 S_i（新しいラベルが網と独立で一様・正が下げる向き）と、式 3・式 3′ の和（float64）。"""
    with torch.no_grad():
        W1, b1 = P[0].detach().double(), P[1].detach().double()
        V, b = P[2].detach().double(), P[3].detach().double()
        Z = ctx.Xd @ W1.T + b1
        A = activ(Z, ctx.act); G = dphi64(Z, ctx.act)
        pi = torch.softmax(A @ V.T + b, dim=1)
        Vc = V - V.mean(0, keepdim=True)
        u = ctx.Kn[:, None] * G
        S = (u * (pi @ Vc)).sum(0)
        E3 = (u * A).sum(0); E3p = (u * (A - A.mean(0, keepdim=True))).sum(0)
        conf = pi.max(1).values.mean()
        return np.stack([S.numpy(), E3.numpy(), E3p.numpy()]), float(conf), float(Vc.norm())


def evaluate(P, ctx, y):
    with torch.no_grad():
        lg = activ(ctx.X @ P[0].T + P[1], ctx.act) @ P[2].T + P[3]
        acc = (lg.argmax(-1) == y).float().mean()
        ce = (lg.logsumexp(-1) - lg.gather(1, y[:, None]).squeeze(1)).mean()
    return float(acc), float(ce)


def run_task(P, opt, ctx, y, Y, order, steps, freeze_body=0, mid_at=None):
    """engine の step() と同じ演算で steps 更新回す。freeze_body > 0 なら最初の freeze_body 更新は隠れ層の勾配を捨てる。"""
    act, X = ctx.act, ctx.X
    upts = [u for u in UPTS if u <= steps]
    if steps not in upts: upts.append(steps)
    units = {}
    push, conf, vnorm = expected_push(P, ctx)
    ev = np.zeros((steps // EVAL_EVERY + 1, 2), np.float32)
    ev[0] = evaluate(P, ctx, y)
    units[0] = unit_stats(P, ctx)
    nb = steps // BIN
    oacc = torch.zeros(steps); oloss = torch.zeros(steps)
    mid = None
    for it in range(steps):
        idx = order[it]
        xb, yb = X[idx], Y[idx]
        z = xb @ P[0].T + P[1]; a = activ(z, act); logits = a @ P[2].T + P[3]
        loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        if it < freeze_body:
            P[0].grad = None; P[1].grad = None
        opt.step()
        with torch.no_grad():
            oloss[it] = loss.detach(); oacc[it] = (logits.argmax(-1) == y[idx]).float().mean()
        u = it + 1
        if mid_at is not None and u == mid_at:
            mid = [q.detach().numpy().copy() for q in P]
        if u in upts and u != 0:
            units[u] = unit_stats(P, ctx)
        if u % EVAL_EVERY == 0:
            ev[u // EVAL_EVERY] = evaluate(P, ctx, y)
    rec = dict(online_acc=oacc[:nb * BIN].reshape(nb, BIN).mean(1).numpy().astype(np.float32),
               online_loss=oloss[:nb * BIN].reshape(nb, BIN).mean(1).numpy().astype(np.float32),
               loss_first=oloss[:min(FIRST, steps)].numpy().astype(np.float32),
               eval=ev, units=np.stack([units[u] for u in upts]), upts=np.array(upts),
               push=push.astype(np.float64), conf0=np.array(conf), vnorm=np.array([vnorm, float((P[2].detach().double() - P[2].detach().double().mean(0)).norm())]))
    return rec, mid


# ---------------- interventions ----------------
def mean_act(P, ctx):
    with torch.no_grad():
        return activ(ctx.X @ P[0].T + P[1], ctx.act).mean(0)       # float32 (H,)


def fix_bias(P, V_old, abar):
    """標本で平均した logit を保つ: b <- b + (V_old - V_new) ā。"""
    with torch.no_grad():
        P[3].add_((V_old - P[2]) @ abar)


def intervene(arm, P, opt, ctx, g, V_init, kind):
    """課題の頭（更新 0 の前）に一度だけ施す。返り値は記録（検査用）。"""
    info = {}
    st = opt.state
    with torch.no_grad():
        V_old = P[2].detach().clone(); b_old = P[3].detach().clone()
        abar = mean_act(P, ctx)
        if arm in ("P0", "P0m"):
            pi = torch.randperm(K, generator=g)
            P[2].copy_(V_old[pi])
            for q in (2,) if arm == "P0m" else (2, 3):
                s = st.get(P[q])
                if s:
                    s["exp_avg"] = s["exp_avg"][pi].clone(); s["exp_avg_sq"] = s["exp_avg_sq"][pi].clone()
            if arm == "P0":
                P[3].copy_(P[3].detach()[pi].clone())
            else:
                fix_bias(P, V_old, abar)
            info["perm"] = pi.numpy()
        elif arm == "P":
            idx = torch.stack([torch.randperm(K, generator=g) for _ in range(H)], dim=1)   # (K, H): 列 j が π_j
            P[2].copy_(V_old.gather(0, idx))
            s = st.get(P[2])
            if s:
                s["exp_avg"] = s["exp_avg"].gather(0, idx).clone(); s["exp_avg_sq"] = s["exp_avg_sq"].gather(0, idx).clone()
            fix_bias(P, V_old, abar)
        elif arm == "Q":
            mu = V_old.mean(0, keepdim=True); Vc = V_old - mu
            V0c = V_init - V_init.mean(0, keepdim=True)
            n0, n1 = V0c.norm(dim=0), Vc.norm(dim=0)
            scale = torch.where(n1 > 0, n0 / n1.clamp_min(1e-30), torch.zeros_like(n1))
            P[2].copy_(mu + Vc * scale[None, :])
            fix_bias(P, V_old, abar)
            info["scale_median"] = float(scale.median())
        elif arm == "R":
            fresh = initial_from(g)
            P[2].copy_(fresh[2]); P[3].copy_(fresh[3])
            st.pop(P[2], None); st.pop(P[3], None)
        elif arm == "H0":
            P[2].zero_(); P[3].zero_()
            st.pop(P[2], None); st.pop(P[3], None)
        elif arm == "H0A":
            P[2].zero_(); P[3].zero_()
            for q in range(4): st.pop(P[q], None)
        elif arm == "A0":
            st.pop(P[0], None); st.pop(P[1], None)
        elif arm == "F200":
            pass
        else:
            raise ValueError(arm)
        # 検査用の記録: 標本で平均した logit のずれ・列の値の集合・中心化した列のノルム
        info["mean_logit_shift"] = float(((abar @ P[2].T + P[3]) - (abar @ V_old.T + b_old)).abs().max())
        info["col_multiset_same"] = bool(torch.equal(V_old.sort(0).values, P[2].detach().sort(0).values))
        c_new = (P[2] - P[2].mean(0, keepdim=True)).norm(dim=0); c_init = (V_init - V_init.mean(0, keepdim=True)).norm(dim=0)
        info["col_norm_rel_to_init_median"] = float((c_new / c_init).median())
    return info


# ---------------- modes ----------------
def git_head():
    try:
        return subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return f"error: {e}"


def run_branch(a):
    act, seed, kind, lr, tasks, steps = a.act, a.seed, a.opt, a.lr, a.tasks, a.steps
    arms = [x for x in a.arms.split(",") if x] if a.arms != "none" else []
    for x in arms: assert x in BRANCH_ARMS, x
    if kind == "sgd": arms = [x for x in arms if x not in ("H0A", "A0")]       # 状態の無い最適化器では意味が無い
    X, subset = load_X(seed); ctx = Ctx(X, act)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P0 = initial(seed, (784, H, K)); V_init = P0[2].clone()
    P = new_params(P0); opt = make_opt(kind, P, lr)
    saved = None
    if kind == "adam" and lr == 1e-3 and (SAVED / f"{act}_K10_s{seed}.npz").exists():
        saved = np.load(SAVED / f"{act}_K10_s{seed}.npz")
        assert np.array_equal(saved["subset"], subset), "subset が保存済みと違う"
        assert maxabs(P, [saved[f"init{q}"] for q in range(4)]) == 0.0, "初期値が保存済みと違う"
    full = steps == ENGINE_STEPS
    res, meta_t = {}, []
    check = dict(end_maxabs=[], mid_maxabs=[], interventions={})
    ends, mids, ys, accs = [], [], [], []
    t0 = time.monotonic()
    for t in range(1, tasks + 1):
        tt = time.monotonic()
        y = torch.randint(K, (N,), generator=glabel)
        order = order_for(gbatch, max(steps, ENGINE_STEPS))[:steps]
        if saved is not None and t <= saved["y"].shape[0]:
            assert np.array_equal(y.numpy(), saved["y"][t - 1]), f"課題 {t} のラベルが保存済みと違う"
        Y = torch.nn.functional.one_hot(y, K).to(torch.float32)

        def put(arm, rec):
            for k, v in rec.items(): res[f"{arm}|t{t:02d}|{k}"] = v

        if t >= 2:
            base = snapshot(P, opt)
            for arm in arms:
                Pb, ob = restore(base, kind, lr)
                if arm == "H0A":
                    ob = make_opt(kind, Pb, lr)
                info = intervene(arm, Pb, ob, ctx, own(f"arm|{arm}|{kind}|{t}", seed), V_init, kind)
                if arm == "F200" and t == 2:
                    w0 = [Pb[0].detach().clone(), Pb[1].detach().clone()]
                rec, _ = run_task(Pb, ob, ctx, y, Y, order, steps, freeze_body=FIRST if arm == "F200" else 0)
                put(arm, rec)
                if t == 2: check["interventions"][arm] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
            del base
        if a.fresh:
            Pf = new_params(initial_from(own(f"fresh_init|{t}", seed)))
            if t == 1:
                check["fresh_vs_engine_init_min_over_params"] = min(float((Pf[q].detach() - P[q].detach()).abs().max()) for q in range(4))
            put("fa", run_task(Pf, make_opt(kind, Pf, lr), ctx, y, Y, order, steps)[0])
        rec, mid = run_task(P, opt, ctx, y, Y, order, steps, mid_at=FIRST)
        put("S", rec)
        if a.save_states:
            ends.append([q.detach().numpy().copy() for q in P]); mids.append(mid)
            ys.append(y.numpy().copy()); accs.append(float(rec["eval"][-1, 0]))
        if saved is not None and full and t <= saved["y"].shape[0]:
            check["end_maxabs"].append(maxabs(P, [saved[f"end{q}"][t - 1] for q in range(4)]))
            check["mid_maxabs"].append(max(float(np.abs(mid[q] - saved[f"mid{q}"][t - 1]).max()) for q in range(4)))
        el = time.monotonic() - tt
        meta_t.append(dict(task=t, elapsed_s=round(el, 2), S_acc_end=float(rec["eval"][-1, 0])))
        print(f"{kind} lr{lr:g} {act} s{seed} task {t:2d} S acc {rec['eval'][-1, 0]:.3f} "
              f"({el:.1f}s, total {time.monotonic() - t0:.0f}s)", flush=True)
    out = Path(a.out); out = out if out.is_absolute() else WORKTREE / out
    out.mkdir(parents=True, exist_ok=True)
    stem = f"branch_{kind}_lr{lr:g}_{act}_s{seed}"
    np.savez_compressed(out / f"{stem}.npz", **res)
    if a.save_states:
        d = dict(subset=subset, y=np.array(ys), acc=np.array(accs), act=act, K=K, seed=seed, opt=kind, lr=lr)
        for j in range(4):
            d[f"init{j}"] = P0[j].numpy(); d[f"end{j}"] = np.stack([e[j] for e in ends]); d[f"mid{j}"] = np.stack([m[j] for m in mids])
        np.savez_compressed(out / f"states_{kind}_lr{lr:g}_{act}_s{seed}.npz", **d)
    meta = dict(mode="branch", act=act, seed=seed, opt=kind, lr=lr, tasks=tasks, steps=steps, arms=arms, fresh=a.fresh,
                torch=torch.__version__, git_head=git_head(), engine_sha256=ENGINE_SHA,
                elapsed_s=round(time.monotonic() - t0, 1), per_task=meta_t, check=check)
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    # 検査（出力を書いた後）
    if saved is not None and full:
        assert all(v == 0.0 for v in check["end_maxabs"]), f"C1 end 不一致 {check['end_maxabs']}"
        assert all(v == 0.0 for v in check["mid_maxabs"]), f"C1 mid 不一致 {check['mid_maxabs']}"
    if a.fresh:
        assert check["fresh_vs_engine_init_min_over_params"] > 0, "C8 fa が engine の初期値と一致"
    print("all checks passed", flush=True)


def run_persist(a):
    act, seed, kind, lr, tasks, steps, arm = a.act, a.seed, a.opt, a.lr, a.tasks, a.steps, a.arm
    assert arm in PERSIST_ARMS, arm
    X, subset = load_X(seed); ctx = Ctx(X, act)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P0 = initial(seed, (784, H, K)); V_init = P0[2].clone()
    P = new_params(P0); opt = make_opt(kind, P, lr)
    res, meta_t = {}, []
    t0 = time.monotonic()
    for t in range(1, tasks + 1):
        tt = time.monotonic()
        y = torch.randint(K, (N,), generator=glabel)
        order = order_for(gbatch, max(steps, ENGINE_STEPS))[:steps]
        Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
        if t >= 2:
            intervene(arm, P, opt, ctx, own(f"persist|{arm}|{kind}|{t}", seed), V_init, kind)
        rec, _ = run_task(P, opt, ctx, y, Y, order, steps, freeze_body=FIRST if (arm == "F200" and t >= 2) else 0)
        for k, v in rec.items(): res[f"{arm}all|t{t:02d}|{k}"] = v
        el = time.monotonic() - tt
        meta_t.append(dict(task=t, elapsed_s=round(el, 2), acc_end=float(rec["eval"][-1, 0])))
        print(f"persist {arm} {kind} {act} s{seed} task {t:2d} acc {rec['eval'][-1, 0]:.3f} ({el:.1f}s)", flush=True)
    out = Path(a.out); out = out if out.is_absolute() else WORKTREE / out
    out.mkdir(parents=True, exist_ok=True)
    stem = f"persist_{arm}_{kind}_lr{lr:g}_{act}_s{seed}"
    np.savez_compressed(out / f"{stem}.npz", **res)
    (out / f"{stem}.json").write_text(json.dumps(dict(mode="persist", arm=arm, act=act, seed=seed, opt=kind, lr=lr, tasks=tasks,
                                                      steps=steps, git_head=git_head(), elapsed_s=round(time.monotonic() - t0, 1),
                                                      per_task=meta_t), indent=1))
    print("done", flush=True)


def run_dial(a):
    """reload_dial_1020 の checkpoint（Adam の状態つき）から、読み出し S・P0・P で新ラベルへ切り替えて 200 更新。"""
    act, seed = a.act, a.seed
    X, subset = load_X(seed); ctx = Ctx(X, act)
    P_init = initial(seed, (784, H, K)); V_init = P_init[2].clone()
    res = {}
    for tag in ("init", "a20", "a50", "a80", "a95", "end"):
        ck = np.load(DIAL / f"ckpt_{act}_s{seed}_{tag}.npz")
        sp = [torch.tensor(ck[f"p{q}"], dtype=torch.float32) for q in range(4)]
        so = []
        for q in range(4):
            so.append({"step": torch.tensor(float(ck[f"step{q}"])), "exp_avg": torch.tensor(ck[f"m{q}"]),
                       "exp_avg_sq": torch.tensor(ck[f"v{q}"])} if float(ck[f"step{q}"]) > 0 else None)
        base = (sp, so)
        for d in range(a.draws):
            y = torch.randint(K, (N,), generator=own(f"dial_label|{tag}|{d}", seed))
            Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
            order = order_for(own(f"dial_batch|{tag}|{d}", seed), FIRST)
            for arm in ("S", "P0", "P", "Q"):
                Pb, ob = restore(base, "adam", 1e-3)
                if arm != "S":
                    intervene(arm, Pb, ob, ctx, own(f"dial_arm|{arm}|{tag}|{d}", seed), V_init, "adam")
                rec, _ = run_task(Pb, ob, ctx, y, Y, order, FIRST)
                for k, v in rec.items(): res[f"{arm}|{tag}|d{d}|{k}"] = v
        print(f"dial {act} s{seed} {tag} done", flush=True)
    out = Path(a.out); out = out if out.is_absolute() else WORKTREE / out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"dial_{act}_s{seed}.npz", **res)
    (out / f"dial_{act}_s{seed}.json").write_text(json.dumps(dict(mode="dial", act=act, seed=seed, draws=a.draws, git_head=git_head()), indent=1))
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["branch", "persist", "dial"])
    ap.add_argument("--act", required=True, choices=["LR", "ELU", "R", "GELU", "SILU"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd", "adaml"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--steps", type=int, default=ENGINE_STEPS)
    ap.add_argument("--arms", default=",".join(BRANCH_ARMS))
    ap.add_argument("--arm", default="P")
    ap.add_argument("--fresh", type=int, default=1)
    ap.add_argument("--save-states", type=int, default=0)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", default="results/stage1_ctrl_1111/raw")
    a = ap.parse_args()
    assert a.steps % BIN == 0 and a.steps % EVAL_EVERY == 0
    {"branch": run_branch, "persist": run_persist, "dial": run_dial}[a.mode](a)


if __name__ == "__main__":
    main()

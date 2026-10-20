#!/usr/bin/env python3
"""chargpt_cl_1019 — LN 付き文字 GPT の継続学習で沈降と喪失を見る。

spec: specs/spec_chargpt_cl_1019.md（vault の v1.1 と追補）。
箱 M: 課題ごとに作り直す乱数の 2 次マルコフ言語（65 記号）。
箱 W: Tiny Shakespeare の固定窓 1,200 個に課題ごとの乱数ラベル（10 値）。

使い方:
  python src/chargpt_cl_1019.py run --box M --arm C3 --seed 0 --tasks 50 --n_per_task 3000 --alpha 0.03 \
      --out results/chargpt_cl_1019/M/C3/seed0
  python src/chargpt_cl_1019.py run --box W --arm C3 --seed 0 --tasks 50 --n_per_task 7500 \
      --out results/chargpt_cl_1019/W/C3/seed0
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RUN_ID = "chargpt_cl_1019"
REPO = Path(__file__).resolve().parents[1]
DATA_W = Path(os.environ.get("CHARGPT_DATA_W", "/home/issan/Projects/claude/proj_004_drift/data/tinyshakespeare/input.txt"))
DATA_W_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

# §2.1 共通のモデル
D, L, H, FF, T, V = 128, 4, 4, 512, 128, 65
N_CLASS = 10
WARMUP = 300

# §3 腕
ARMS = {
    "C3": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3),
    "C4": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-4),
    "B3": dict(norm="ln", ln_bias=True, lin_bias=True, lr=1e-3),
    "N4": dict(norm="none", ln_bias=False, lin_bias=False, lr=1e-4),
}

# §4.1 時点
WITHIN_TASKS = (1, 2, 10, 11, 25, 26, 49, 50)
FRESH_TASKS = (10, 25, 50)
CKPT_TASKS = (1, 10, 25, 50)

# §2.2 / §2.3 の固定値
M_BATCH, M_SEQ = 32, T + 1
M_PROBE, M_EVAL = 64, 256
W_N_WIN, W_BATCH, W_PROBE = 1200, 16, 256
M_FIRST, W_FIRST = 500, 1000          # §4.2 の「課題の最初」の更新数
M_CHUNK = 250                          # 系列を作る単位（更新数）。課題の頭から同じ順に作る


def setup_torch():
    torch.set_num_threads(int(os.environ.get("CHARGPT_THREADS", "2")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------- 乱数の stream（§3 seed）
def stream_seed(seed, stream, *keys):
    s = "/".join([RUN_ID, str(seed), stream] + [str(k) for k in keys])
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def cpu_gen(seed, stream, *keys):
    return torch.Generator().manual_seed(stream_seed(seed, stream, *keys))


def np_gen(seed, stream, *keys):
    return np.random.Generator(np.random.PCG64(stream_seed(seed, stream, *keys)))


# ---------------------------------------------------------------- モデル
def rope_tables():
    hd = D // H
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).double() / hd))
    f = torch.outer(torch.arange(T).double(), inv)
    return f.cos().float()[None, None], f.sin().float()[None, None]


def rope(x, c, s):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)


def make_norm(arm):
    a = ARMS[arm]
    return nn.LayerNorm(D, bias=a["ln_bias"]) if a["norm"] == "ln" else nn.Identity()


class Attn(nn.Module):
    def __init__(s, arm):
        super().__init__()
        b = ARMS[arm]["lin_bias"]
        s.qkv = nn.Linear(D, 3 * D, bias=b)
        s.o = nn.Linear(D, D, bias=b)
        c, sn = rope_tables()
        s.register_buffer("cos", c, persistent=False)
        s.register_buffer("sin", sn, persistent=False)
        s.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1), persistent=False)

    def forward(s, x):
        B, Tt, C = x.shape
        q, k, v = s.qkv(x).split(C, 2)
        q, k, v = [t.view(B, Tt, H, C // H).transpose(1, 2) for t in (q, k, v)]
        q, k = rope(q, s.cos[..., :Tt, :], s.sin[..., :Tt, :]), rope(k, s.cos[..., :Tt, :], s.sin[..., :Tt, :])
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // H))
        att = att.masked_fill(s.mask[:Tt, :Tt], float("-inf")).softmax(-1)
        return s.o((att @ v).transpose(1, 2).reshape(B, Tt, C))


class MLP(nn.Module):
    def __init__(s, arm):
        super().__init__()
        b = ARMS[arm]["lin_bias"]
        s.up = nn.Linear(D, FF, bias=b)
        s.down = nn.Linear(FF, D, bias=b)

    def forward(s, y, cap=None):
        z = s.up(y)
        if cap is not None:
            cap.append(z)
        return s.down(F.relu(z))


class Block(nn.Module):
    def __init__(s, arm):
        super().__init__()
        s.n1, s.n2 = make_norm(arm), make_norm(arm)
        s.attn, s.mlp = Attn(arm), MLP(arm)
        s.sc = 1.0 / math.sqrt(2 * L)

    def forward(s, x, cap=None):
        x = x + s.sc * s.attn(s.n1(x))
        if cap is not None:
            cap.append(x)                 # FFN 前の LN の入力
        return x + s.mlp(s.n2(x), cap)


class GPT(nn.Module):
    def __init__(s, arm, box):
        super().__init__()
        s.arm, s.box = arm, box
        s.wte = nn.Embedding(V, D)
        s.h = nn.ModuleList([Block(arm) for _ in range(L)])
        s.nf = make_norm(arm)
        s.head = nn.Linear(D, V, bias=False)
        s.wte.weight = s.head.weight      # tied embedding
        if box == "W":
            s.readout = nn.Linear(D, N_CLASS, bias=ARMS[arm]["lin_bias"])

    def body(s, x, cap=None):
        h = s.wte(x)
        for b in s.h:
            h = b(h, cap)
        return s.nf(h)

    def forward(s, x, cap=None):
        h = s.body(x, cap)
        if s.box == "M":
            return s.head(h)
        return s.readout(h.mean(1))


@torch.no_grad()
def init_params(model, seed):
    """PyTorch 既定の分布で、パラメータ名ごとの stream から引く（腕の間で共有できる部分を同じにする）。"""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            bound = 1.0 / math.sqrt(mod.in_features)
            for pn in ("weight", "bias"):
                p = getattr(mod, pn)
                if p is None:
                    continue
                g = cpu_gen(seed, "init", f"{name}.{pn}")
                p.copy_(torch.empty(p.shape).uniform_(-bound, bound, generator=g))
        elif isinstance(mod, nn.LayerNorm):
            mod.weight.fill_(1.0)
            if mod.bias is not None:
                mod.bias.zero_()


def build_model(arm, box, seed, device):
    m = GPT(arm, box)
    init_params(m, seed)
    return m.to(device)


def state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- 箱 M（§2.2）
def stationary(P):
    """(a,b) の組を状態とする連鎖の定常分布。周期を避けるため遅延連鎖でべき乗法（定常分布は同じ）。"""
    pi = np.full((V, V), 1.0 / (V * V))
    for it in range(1, 1_000_001):
        new = 0.5 * pi + 0.5 * np.einsum("ab,abc->bc", pi, P)
        d = np.abs(new - pi).sum()
        pi = new
        if d < 1e-14:
            break
    pi = pi / pi.sum()
    return pi, it


class ChainM:
    def __init__(s, seed, task, alpha):
        rng = np_gen(seed, "rule", task)
        P = rng.dirichlet(alpha * np.ones(V), size=V * V)          # (4225, 65)
        s.P = P.reshape(V, V, V)
        s.pi, s.pi_iters = stationary(s.P)
        with np.errstate(divide="ignore", invalid="ignore"):
            hrow = -np.where(s.P > 0, s.P * np.log(np.where(s.P > 0, s.P, 1.0)), 0.0).sum(2)
        s.hrow = hrow
        s.H = float((s.pi * hrow).sum())
        s.rowsum_err = float(np.abs(P.sum(1) - 1.0).max())
        # 系列は CPU の乱数で作る（CUDA の乱数は GPU の機種で値が変わり得るため。追補 1）
        s.cum = torch.from_numpy(np.cumsum(s.P.reshape(V * V, V), 1))
        s.pi_cum = torch.from_numpy(np.cumsum(s.pi.reshape(-1)))
        s.logP = torch.from_numpy(np.log(np.maximum(s.P.reshape(V * V, V), 1e-300)))

    def sample(s, n, gen):
        """長さ 129 の系列 n 本（CPU）。最初の 2 記号は π から、以後 p(·|a,b) から。gen は CPU の Generator。"""
        dev = s.cum.device
        u0 = torch.rand(n, 1, generator=gen, device=dev, dtype=torch.float64)
        st = torch.searchsorted(s.pi_cum, u0.squeeze(1), right=True).clamp_(max=V * V - 1)
        seq = torch.empty(n, M_SEQ, dtype=torch.long, device=dev)
        seq[:, 0], seq[:, 1] = st // V, st % V
        U = torch.rand(n, M_SEQ - 2, generator=gen, device=dev, dtype=torch.float64).T.contiguous()
        for j in range(2, M_SEQ):
            ctx = seq[:, j - 2] * V + seq[:, j - 1]
            seq[:, j] = torch.searchsorted(s.cum[ctx], U[j - 2].unsqueeze(1), right=True).squeeze(1).clamp_(max=V - 1)
        return seq

    def true_ce(s, seq):
        """真の連鎖そのものの予測の交差エントロピー（系列ごとの平均、目標 s_2..s_128）。"""
        seq = seq.cpu()
        ctx = seq[:, :-2] * V + seq[:, 1:-1]
        return -s.logP[ctx, seq[:, 2:]].mean(1)


def task_data_M(chain, seed, task, n):
    gen = cpu_gen(seed, "data", task)
    out = torch.empty(n, M_BATCH, M_SEQ, dtype=torch.uint8)
    for c0 in range(0, n, M_CHUNK):
        c1 = min(n, c0 + M_CHUNK)
        out[c0:c1] = chain.sample((c1 - c0) * M_BATCH, gen).view(c1 - c0, M_BATCH, M_SEQ).to(torch.uint8)
    return out


def probes_M(chain, seed, task):
    chain_probe = chain.sample(M_PROBE, cpu_gen(seed, "probe", task, "chain"))[:, :T]
    eval_seq = chain.sample(M_EVAL, cpu_gen(seed, "probe", task, "eval"))
    return chain_probe, eval_seq


def task_cpu_M(seed, task, n, alpha):
    """課題 t の規則・batch 列・probe を CPU で作る（裏のスレッドで先に作れるように分けた）。"""
    chain = ChainM(seed, task, alpha)
    data = task_data_M(chain, seed, task, n)
    probe, eval_seq = probes_M(chain, seed, task)
    return dict(chain=chain, data=data, probe=probe, eval_seq=eval_seq, data_sha=tensor_sha(data),
                probe_sha=tensor_sha(probe), rule_sha=hashlib.sha256(chain.P.tobytes()).hexdigest())


def uniform_probe_M(seed, device):
    g = cpu_gen(seed, "probe", "uniform")
    return torch.randint(0, V, (M_PROBE, T), generator=g).to(device)


# ---------------------------------------------------------------- 箱 W（§2.3）
class WindowsW:
    def __init__(s, seed, device, path=DATA_W):
        raw = Path(path).read_bytes()
        s.data_sha = hashlib.sha256(raw).hexdigest()
        if s.data_sha != DATA_W_SHA256:
            raise SystemExit(f"Tiny Shakespeare の sha256 が違う: {s.data_sha}")
        text = raw.decode("utf-8")
        chars = sorted(set(text))
        assert len(chars) == V
        stoi = {c: i for i, c in enumerate(chars)}
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n90 = int(0.9 * len(ids))
        g = cpu_gen(seed, "rule", "windows")
        s.offset = int(torch.randint(0, T, (1,), generator=g))
        nblk = (n90 - s.offset) // T
        s.blocks = torch.randperm(nblk, generator=g)[:W_N_WIN]
        s.starts = s.offset + s.blocks * T                     # 重ならない窓の先頭
        s.windows = torch.stack([ids[a:a + T] for a in s.starts.tolist()]).to(device)
        g = cpu_gen(seed, "probe", "W")
        s.probe_idx = torch.randperm(W_N_WIN, generator=g)[:W_PROBE].to(device)
        s.seed, s.device = seed, device

    def labels(s, task):
        g = cpu_gen(s.seed, "rule", task)
        return torch.randint(0, N_CLASS, (W_N_WIN,), generator=g).to(s.device)

    def batch_idx(s, task, n):
        g = cpu_gen(s.seed, "data", task)
        per = W_N_WIN // W_BATCH
        ep = -(-n // per)
        idx = torch.cat([torch.randperm(W_N_WIN, generator=g) for _ in range(ep)])[:n * W_BATCH]
        return idx.view(n, W_BATCH).to(s.device)


# ---------------------------------------------------------------- 学習の 1 更新（compile ＋ CUDA グラフ）
class Engine:
    """1 更新 = 更新前のバッチの損失（オンライン成績）を記録して Adam で 1 歩。

    graph=True: eager の 1 更新をそのまま CUDA グラフに取り込み、入力・lr は静的な tensor に書き込む。
    graph=False: そのままの eager（検査の参照）。
    torch.compile は使わない（追補 0 §A: RoPE を含むモデル全体で勾配が float64 の参照から 3e−5〜3e−4 ずれた）。
    """

    def __init__(s, model, arm, box, graph=True):
        s.m, s.box, s.graph = model, box, graph
        dev = next(model.parameters()).device
        s.base_lr = ARMS[arm]["lr"]
        B = M_BATCH if box == "M" else W_BATCH
        s.x = torch.zeros(B, T, dtype=torch.long, device=dev)
        s.y = torch.zeros((B, T) if box == "M" else (B,), dtype=torch.long, device=dev)
        if graph:
            s.lr = torch.tensor(s.base_lr, dtype=torch.float32, device=dev)
            s.opt = torch.optim.Adam(model.parameters(), lr=s.lr, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0, foreach=True, capturable=True)
            s.fn = s._loss
            s._capture()
        else:
            s.opt = torch.optim.Adam(model.parameters(), lr=s.base_lr, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0, foreach=True)
            s.fn = s._loss

    def _loss(s, x, y):
        out = s.m(x)
        if s.box == "M":
            loss = F.cross_entropy(out[:, 1:].reshape(-1, V), y[:, 1:].reshape(-1))
            return loss, loss
        loss = F.cross_entropy(out, y)
        return loss, (out.argmax(1) == y).float().mean()

    def _step_body(s):
        loss, aux = s.fn(s.x, s.y)
        loss.backward()
        s.opt.step()
        return loss.detach(), aux.detach()

    def _capture(s):
        params = [p for p in s.m.parameters()]
        snap = [p.detach().clone() for p in params]
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                s.opt.zero_grad(set_to_none=True)
                s._step_body()
        torch.cuda.current_stream().wait_stream(st)
        s.opt.zero_grad(set_to_none=True)
        s.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.g):
            s.s_loss, s.s_aux = s._step_body()
        # 暖機で動いた重みと Adam の状態を初期に戻す（グラフが指す記憶はそのまま）
        with torch.no_grad():
            for p, q in zip(params, snap):
                p.copy_(q)
            for p in params:
                stt = s.opt.state[p]
                stt["exp_avg"].zero_()
                stt["exp_avg_sq"].zero_()
                stt["step"].zero_()
        torch.cuda.synchronize()

    def set_lr(s, lr):
        if s.graph:
            s.lr.fill_(lr)
        else:
            for gr in s.opt.param_groups:
                gr["lr"] = lr

    def step(s, x, y, buf_loss, buf_aux, u):
        s.x.copy_(x)
        s.y.copy_(y)
        if s.graph:
            s.g.replay()
            buf_loss[u].copy_(s.s_loss)
            buf_aux[u].copy_(s.s_aux)
        else:
            s.opt.zero_grad(set_to_none=True)
            loss, aux = s._step_body()
            buf_loss[u].copy_(loss)
            buf_aux[u].copy_(aux)


# ---------------------------------------------------------------- 測定（§4.1）
@torch.no_grad()
def measure(model, x):
    """FFN の up 出力 z の unit ごとの量と平均の分解。返り値は層ごとの dict（numpy）。"""
    cap = []
    model.body(x, cap)
    out = []
    for li, blk in enumerate(model.h):
        xin, z = cap[2 * li], cap[2 * li + 1]
        z2 = z.reshape(-1, FF)
        z64 = z2.double()
        mu = z64.mean(0)
        sd = z64.std(0)
        q = (z2 < 0).double().mean(0)
        dead = (z2 > 0).sum(0) == 0
        n2 = blk.n2
        if isinstance(n2, nn.LayerNorm):
            xhat = F.layer_norm(xin, (D,), eps=n2.eps).reshape(-1, D).double()
            gam = n2.weight.double()
            beta = n2.bias.double() if n2.bias is not None else None
        else:
            xhat = xin.reshape(-1, D).double()
            gam, beta = torch.ones(D, dtype=torch.float64, device=x.device), None
        mhat = xhat.mean(0)
        W = blk.mlp.up.weight.double()
        gm = gam * mhat
        t1 = W @ gm
        t2 = W @ beta if beta is not None else torch.zeros_like(t1)
        t3 = blk.mlp.up.bias.double() if blk.mlp.up.bias is not None else torch.zeros_like(t1)
        cos = t1 / (W.norm(dim=1) * gm.norm() + 1e-30)
        r = mu / (sd + 1e-12)
        rq = torch.quantile(r, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, device=r.device))
        agg = dict(neg_frac=float((z2 < 0).double().mean()), units_mean_neg=float((mu < 0).double().mean()),
                   r_q25=float(rq[0]), r_med=float(rq[1]), r_q75=float(rq[2]),
                   sunk=float((q >= 0.95).double().mean()), dead=float(dead.double().mean()),
                   decomp_err=float((mu - (t1 + t2 + t3)).abs().max()),
                   mu_absmean=float(mu.abs().mean()))
        units = dict(mu=mu, sd=sd, q=q, t_wgm=t1, t_wbeta=t2, t_b=t3, cos=cos)
        out.append(dict(agg=agg, units={k: v.float().cpu().numpy() for k, v in units.items()},
                        dead=dead.cpu().numpy()))
    return out


@torch.no_grad()
def weight_stats(model):
    rn, rm = [], []
    for blk in model.h:
        W = blk.mlp.up.weight.double()
        rn.append(W.norm(dim=1).float().cpu().numpy())
        rm.append(W.mean(1).float().cpu().numpy())
    return np.stack(rn), np.stack(rm)


@torch.no_grad()
def eval_M(model, eval_seq, H):
    lg = model(eval_seq[:, :T])
    ce = F.cross_entropy(lg[:, 1:].reshape(-1, V), eval_seq[:, 2:].reshape(-1))
    return float(ce) - H


@torch.no_grad()
def eval_W(model, W, labels):
    correct, loss = 0, 0.0
    for c0 in range(0, W_N_WIN, 300):
        lg = model(W.windows[c0:c0 + 300])
        correct += int((lg.argmax(1) == labels[c0:c0 + 300]).sum())
        loss += float(F.cross_entropy(lg, labels[c0:c0 + 300], reduction="sum"))
    return correct / W_N_WIN, loss / W_N_WIN


def within_points(n, full):
    if not full:
        return sorted({0, min(100, n), n})
    pts = {0, 10, 25, 50, 100, 200, 400} | set(range(500, n + 1, 500)) | {n}
    return sorted(p for p in pts if p <= n)


# ---------------------------------------------------------------- 1 走
def git_info():
    try:
        h = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "src/"],
                                        text=True).strip().splitlines()
    except Exception:
        h, dirty = "unknown", []
    return h, dirty


class TaskCtx:
    """課題 t の規則・データ・probe（継続と fresh で同じものを使う）。"""

    def __init__(s, box, seed, task, n, device, alpha=None, W=None, cpu=None):
        s.box, s.task, s.n = box, task, n
        if box == "M":
            c = cpu if cpu is not None else task_cpu_M(seed, task, n, alpha)
            s.chain, s.H = c["chain"], c["chain"].H
            s.data = c["data"].to(device)
            s.probe, s.eval_seq = c["probe"].to(device), c["eval_seq"].to(device)
            s.data_sha, s.rule_sha, s.probe_sha = c["data_sha"], c["rule_sha"], c["probe_sha"]
        else:
            s.W = W
            s.H = 0.0
            s.labels = W.labels(task)
            s.idx = W.batch_idx(task, n)
            s.data_sha = tensor_sha(s.idx)
            s.rule_sha = tensor_sha(s.labels)
            s.probe_sha = tensor_sha(W.probe_idx)

    def batch(s, u):
        if s.box == "M":
            b = s.data[u]
            return b[:, :T], b[:, 1:]
        i = s.idx[u]
        return s.W.windows[i], s.labels[i]


def run_task(model, eng, ctx, *, warmup_from, measure_cb=None, points=()):
    """課題 1 つ。warmup_from: この課題の頭の時点での通算更新数（< WARMUP なら線形 warmup）。"""
    dev = next(model.parameters()).device
    n = ctx.n
    buf_loss = torch.zeros(n, device=dev)
    buf_aux = torch.zeros(n, device=dev)
    pts = set(points)
    lr_now = None
    for u in range(n):
        if u in pts and measure_cb is not None:
            measure_cb(u)
        g = warmup_from + u
        lr = eng.base_lr * min(1.0, (g + 1) / WARMUP)
        if lr != lr_now:
            eng.set_lr(lr)
            lr_now = lr
        x, y = ctx.batch(u)
        eng.step(x, y, buf_loss, buf_aux, u)
    if n in pts and measure_cb is not None:
        measure_cb(n)
    return buf_loss.cpu().numpy(), buf_aux.cpu().numpy()


def summarize_online(box, loss, aux, H):
    first = M_FIRST if box == "M" else W_FIRST
    if box == "M":
        ex = loss.astype(np.float64) - H
        return dict(online_first=float(ex[:first].mean()), online_all=float(ex.mean()))
    acc = aux.astype(np.float64)
    return dict(online_first=float(acc[:first].mean()), online_all=float(acc.mean()),
                online_loss_all=float(loss.astype(np.float64).mean()))


def run(args):
    setup_torch()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ckpts").mkdir(exist_ok=True)
    box, arm, seed, n_tasks, n = args.box, args.arm, args.seed, args.tasks, args.n_per_task
    graph = args.engine == "graph"
    t_start = time.time()
    gh, dirty = git_info()            # 走ったコードの版は開始時に読む（走っている間の commit で HEAD が進むため）
    W = WindowsW(seed, dev) if box == "W" else None
    probe_u = uniform_probe_M(seed, dev) if box == "M" else None

    model = build_model(arm, box, seed, dev)
    init_sha = state_sha(model)
    if args.ckpt:
        torch.save(model.state_dict(), out / "ckpts" / "init.pt")
    eng = Engine(model, arm, box, graph=graph)
    assert state_sha(model) == init_sha, "グラフの取り込みで初期値が変わった"

    within = set(args.within_tasks) if args.within_tasks is not None else set(WITHIN_TASKS)
    fresh_tasks = set(args.fresh_tasks) if args.fresh_tasks is not None else set(FRESH_TASKS)
    measure_on = not args.no_measure

    rows, layer_rows, fresh_rows, eval_rows, task_secs = [], [], [], [], []
    online = dict(loss=np.zeros((n_tasks, n), np.float32), aux=np.zeros((n_tasks, n), np.float32))
    unit_keys = ("mu", "sd", "q", "t_wgm", "t_wbeta", "t_b", "cos")
    probes = ["chain", "uniform"] if box == "M" else ["windows"]
    U = {f"{p}_{k}": np.zeros((n_tasks, L, FF), np.float32) for p in probes for k in unit_keys}
    U.update({f"{p}_dead": np.zeros((n_tasks, L, FF), bool) for p in probes})
    U.update({f"{p}_{k}_u{uu}": np.zeros((n_tasks, L, FF), np.float32)
              for p in probes for k in ("mu", "sd") for uu in (0, 100)})
    U["rownorm"] = np.zeros((n_tasks, L, FF), np.float32)
    U["rowmean"] = np.zeros((n_tasks, L, FF), np.float32)
    WT = {}
    H_list, provenance_tasks = [], []

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1) if box == "M" else None
    fut = pool.submit(task_cpu_M, seed, 1, n, args.alpha) if pool else None
    for t in range(1, n_tasks + 1):
        ti = t - 1
        cpu = fut.result() if fut is not None else None
        if pool is not None and t < n_tasks:
            fut = pool.submit(task_cpu_M, seed, t + 1, n, args.alpha)     # 次の課題の系列を裏で作る
        ctx = TaskCtx(box, seed, t, n, dev, alpha=args.alpha, W=W, cpu=cpu)
        H_list.append(ctx.H)
        full = t in within
        pts = within_points(n, full) if measure_on else [n]
        wt_rec = dict(points=[], **{f"{k}": [] for k in ("mu", "sd", "q")}, eval=[], eval2=[])

        def probe_inputs():
            if box == "M":
                return dict(chain=ctx.probe, uniform=probe_u)
            return dict(windows=W.windows[W.probe_idx])

        def measure_cb(u):
            ev = eval_M(model, ctx.eval_seq, ctx.H) if box == "M" else eval_W(model, W, ctx.labels)
            ev1, ev2 = (ev, float("nan")) if box == "M" else ev
            for p, xp in probe_inputs().items():
                res = measure(model, xp)
                for li, r in enumerate(res):
                    layer_rows.append(dict(task=t, u=u, probe=p, layer=li, **r["agg"]))
                    if u == n:
                        for k in unit_keys:
                            U[f"{p}_{k}"][ti, li] = r["units"][k]
                        U[f"{p}_dead"][ti, li] = r["dead"]
                    if u in (0, 100) and u != n:
                        U[f"{p}_mu_u{u}"][ti, li] = r["units"]["mu"]
                        U[f"{p}_sd_u{u}"][ti, li] = r["units"]["sd"]
                if full and p == probes[0]:
                    wt_rec["mu"].append(np.stack([r["units"]["mu"] for r in res]))
                    wt_rec["sd"].append(np.stack([r["units"]["sd"] for r in res]))
                    wt_rec["q"].append(np.stack([r["units"]["q"] for r in res]))
            if full:
                wt_rec["points"].append(u)
                wt_rec["eval"].append(ev1)
                wt_rec["eval2"].append(ev2)
            eval_rows.append(dict(task=t, u=u, eval=ev1, eval_loss=ev2))

        tt0 = time.time()
        loss, aux = run_task(model, eng, ctx, warmup_from=ti * n, measure_cb=measure_cb, points=pts)
        online["loss"][ti], online["aux"][ti] = loss, aux
        rn, rm = weight_stats(model)
        U["rownorm"][ti], U["rowmean"][ti] = rn, rm
        endm = eval_M(model, ctx.eval_seq, ctx.H) if box == "M" else eval_W(model, W, ctx.labels)
        row = dict(task=t, n=n, H=ctx.H, **summarize_online(box, loss, aux, ctx.H),
                   end_metric=endm if box == "M" else endm[0],
                   end_loss=float("nan") if box == "M" else endm[1],
                   data_sha=ctx.data_sha[:16], rule_sha=ctx.rule_sha[:16])
        rows.append(row)
        task_secs.append(time.time() - tt0)
        provenance_tasks.append(dict(task=t, data_sha=ctx.data_sha, rule_sha=ctx.rule_sha, H=ctx.H,
                                     probe_sha=ctx.probe_sha,
                                     pi_iters=getattr(getattr(ctx, "chain", None), "pi_iters", None)))
        if full and wt_rec["points"]:
            for k in ("mu", "sd", "q"):
                WT[f"t{t}_{k}"] = np.stack(wt_rec[k]).astype(np.float32)
            WT[f"t{t}_points"] = np.array(wt_rec["points"])
            WT[f"t{t}_eval"] = np.array(wt_rec["eval"])
            WT[f"t{t}_eval2"] = np.array(wt_rec["eval2"])
        if args.ckpt and t in CKPT_TASKS:
            torch.save(model.state_dict(), out / "ckpts" / f"task{t}.pt")
        print(f"[{box}/{arm}/s{seed}] task {t}/{n_tasks} online_all {row['online_all']:.4f} "
              f"end {row['end_metric']:.4f} {task_secs[-1]:.0f}s", flush=True)

        if t in fresh_tasks:
            fm = build_model(arm, box, seed, dev)
            assert state_sha(fm) == init_sha
            fe = Engine(fm, arm, box, graph=graph)
            fl, fa = run_task(fm, fe, ctx, warmup_from=0)
            fend = eval_M(fm, ctx.eval_seq, ctx.H) if box == "M" else eval_W(fm, W, ctx.labels)
            frow = dict(task=t, n=n, H=ctx.H, **summarize_online(box, fl, fa, ctx.H),
                        end_metric=fend if box == "M" else fend[0], data_sha=ctx.data_sha[:16])
            for p, xp in probe_inputs().items():
                for li, r in enumerate(measure(fm, xp)):
                    frow.update({f"{p}_l{li}_{k}": v for k, v in r["agg"].items()})
            fresh_rows.append(frow)
            np.savez_compressed(out / f"fresh_online_t{t}.npz", loss=fl, aux=fa)
            print(f"[{box}/{arm}/s{seed}] fresh t{t} online_all {frow['online_all']:.4f}", flush=True)
            del fm, fe
            torch.cuda.empty_cache() if dev.type == "cuda" else None
        del ctx

    import csv

    def wcsv(path, rws):
        if not rws:
            return
        keys = list(dict.fromkeys(k for r in rws for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rws:
                w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})

    wcsv(out / "per_task.csv", rows)
    wcsv(out / "layer_stats.csv", layer_rows)
    wcsv(out / "fresh.csv", fresh_rows)
    wcsv(out / "eval_trace.csv", eval_rows)
    np.savez_compressed(out / "unit_stats.npz", **U)
    np.savez_compressed(out / "within_task.npz", **WT)
    np.savez_compressed(out / "online.npz", **online)
    gh_end, _ = git_info()
    import socket
    prov = dict(run_id=RUN_ID, host=socket.gethostname(), numpy=np.__version__, python=sys.version.split()[0],
                box=box, arm=arm, arm_cfg=ARMS[arm], seed=seed, tasks=n_tasks, n_per_task=n,
                alpha=args.alpha, engine=args.engine, git_hash=gh, dirty_src=dirty, git_hash_at_end=gh_end,
                torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                init_sha=init_sha, final_sha=state_sha(model), tasks_info=provenance_tasks,
                data_W_sha256=W.data_sha if W else None,
                W_window_starts=W.starts.tolist() if W else None,
                W_probe_idx=W.probe_idx.cpu().tolist() if W else None,
                probe_uniform_sha=tensor_sha(probe_u) if box == "M" else None,
                stream_seed_example=dict(init=stream_seed(seed, "init", "h.0.mlp.up.weight"),
                                         rule_t1=stream_seed(seed, "rule", 1),
                                         data_t1=stream_seed(seed, "data", 1)),
                within_tasks=sorted(within), fresh_tasks=sorted(fresh_tasks),
                wall_sec=time.time() - t_start, task_secs=task_secs, argv=sys.argv)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"[{box}/{arm}/s{seed}] done {time.time() - t_start:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--box", choices=["M", "W"], required=True)
    r.add_argument("--arm", choices=list(ARMS), required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--tasks", type=int, default=50)
    r.add_argument("--n_per_task", type=int, required=True)
    r.add_argument("--alpha", type=float, default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--engine", choices=["graph", "eager"], default="graph")
    r.add_argument("--within_tasks", type=int, nargs="*", default=None)
    r.add_argument("--fresh_tasks", type=int, nargs="*", default=None)
    r.add_argument("--no_measure", action="store_true")
    r.add_argument("--ckpt", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run":
        if a.box == "M" and a.alpha is None:
            ap.error("箱 M は --alpha が要る")
        run(a)


if __name__ == "__main__":
    main()

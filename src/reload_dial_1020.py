#!/usr/bin/env python3
"""reload_dial_1020: 課題 1 を当てはめた自信と、切替の押しの用量反応（1 隠れ層 RL-MNIST）。

箱と乱数列は obsidian-research-data/push_direction_proof_0930/gen_states.py
（sha256 4571091d4c02ef6a3d74457a62ad128f2ae177525d8a48d47f2ed70b0fec3f30）と同じ:
784-100-10、MNIST raw/255・subset 1,200、乱数ラベル K = 10、Adam 1e-3 (0.9, 0.999, 1e-8)・batch 16・CPU 1 スレッド・float32。

段 replay: 課題 1 を 4,000 更新回す。10 更新ごとに 1,200 枚の訓練精度を測り、初めて 0.20・0.50・0.80・0.95 以上に
  なった更新と、初期値・課題末で θ と Adam 状態を保存する。200 更新後と課題末の θ を保存済みの mid[0]・end[0] と照合する。
  続けて engine の乱数列で課題 2 の最初の 200 更新を回し、保存済みの mid[1] と照合する（Adam 状態を引き継ぐ切替の検査）。
  変異対照: 同じ 200 更新を Adam 状態を捨てて回すと mid[1] と一致しないこと。
段 switch: 各 checkpoint から、新ラベル（独立・一様）を draw ごとの乱数列で当て、θ と Adam 状態を引き継いで 200 更新回す。
  1・2・5・10・20・50・100・200 更新後の位置 m（float64）と開いた割合 p を全 unit で保存する。
"""
import argparse, copy, gzip, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
SAVED = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/states")
N, BATCH, H, K = 1200, 16, 100, 10
LR, BETAS, EPS = 1e-3, (0.9, 0.999), 1e-8
STEPS = 4000
TARGETS = [0.20, 0.50, 0.80, 0.95]
SW_STEPS = 200
SW_SAVE = [1, 2, 5, 10, 20, 50, 100, 200]
SQRT2 = math.sqrt(2.0)


def stream(role, seed):                       # gen_states.py と同一（engine の乱数列）
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def dial_stream(role, seed):                  # この走だけの乱数列（engine の列と衝突しない）
    h = hashlib.sha256(f"reload_dial_1020|{role}|{seed}".encode()).digest()
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


def replay(act, seed, out):
    X, subset = load_X(seed)
    saved = np.load(SAVED / f"{act}_K10_s{seed}.npz")
    assert np.array_equal(saved["subset"], subset), "subset が保存済みと違う"
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(seed, (784, H, K))]
    optim = torch.optim.Adam(P, lr=LR, betas=BETAS, eps=EPS)
    check = {"init_equal": all(np.array_equal(P[q].detach().numpy(), saved[f"init{q}"]) for q in range(4))}
    y1 = torch.randint(K, (N,), generator=glabel); Y1 = torch.nn.functional.one_hot(y1, K).to(torch.float32)
    assert np.array_equal(y1.numpy(), saved["y"][0]), "課題 1 のラベルが保存済みと違う"
    order = order_for(gbatch, STEPS)
    ckpts = {"init": snap(P, optim, 0, accuracy(P, X, y1, act))}
    todo = list(TARGETS); trace = []
    for it in range(STEPS):
        step(P, optim, X, Y1, order[it], act)
        if it + 1 == 200:
            check["mid0_maxabs"] = max(float(np.abs(P[q].detach().numpy() - saved[f"mid{q}"][0]).max()) for q in range(4))
        if (it + 1) % 10 == 0:
            acc = accuracy(P, X, y1, act); trace.append((it + 1, acc))
            while todo and acc >= todo[0]:
                ckpts[f"a{int(round(100 * todo[0]))}"] = snap(P, optim, it + 1, acc); todo.pop(0)
    ckpts["end"] = snap(P, optim, STEPS, accuracy(P, X, y1, act))
    check["end0_maxabs"] = max(float(np.abs(P[q].detach().numpy() - saved[f"end{q}"][0]).max()) for q in range(4))
    check["targets_not_reached"] = todo

    # 切替の検査: 課題 2 の最初の 200 更新を engine の乱数列で回し、mid[1] と照合（Adam 状態を引き継ぐ）
    y2 = torch.randint(K, (N,), generator=glabel); Y2 = torch.nn.functional.one_hot(y2, K).to(torch.float32)
    assert np.array_equal(y2.numpy(), saved["y"][1]), "課題 2 のラベルが保存済みと違う"
    order2 = order_for(gbatch, STEPS)
    Pk, ok = restore(ckpts["end"])
    for it in range(200): step(Pk, ok, X, Y2, order2[it], act)
    check["mid1_maxabs"] = max(float(np.abs(Pk[q].detach().numpy() - saved[f"mid{q}"][1]).max()) for q in range(4))
    ck_fresh = dict(ckpts["end"]); ck_fresh.update({f"step{q}": np.array(0.0) for q in range(4)})   # 変異: Adam 状態を捨てる
    Pm, om = restore(ck_fresh)
    for it in range(200): step(Pm, om, X, Y2, order2[it], act)
    check["mid1_mutant_maxabs"] = max(float(np.abs(Pm[q].detach().numpy() - saved[f"mid{q}"][1]).max()) for q in range(4))

    out.mkdir(parents=True, exist_ok=True)
    for name, ck in ckpts.items():
        np.savez_compressed(out / f"ckpt_{act}_s{seed}_{name}.npz", **ck)
    meta = dict(act=act, seed=seed, torch=torch.__version__, check=check, trace=trace,
                ckpts={name: dict(update=int(ck["update"]), acc=float(ck["acc"])) for name, ck in ckpts.items()})
    (out / f"replay_{act}_s{seed}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(dict(act=act, seed=seed, check=check, ckpts=meta["ckpts"])), flush=True)


def switch(act, seed, ckdir, out, draws):
    X, subset = load_X(seed)
    xbar = X.double().mean(0)
    res = {}
    for f in sorted(ckdir.glob(f"ckpt_{act}_s{seed}_*.npz")):
        name = f.stem.split("_")[-1]; ck = dict(np.load(f))
        for j in range(draws):
            g_lab, g_bat = dial_stream(f"labels|{j}", seed), dial_stream(f"batch|{j}", seed)
            y = torch.randint(K, (N,), generator=g_lab); Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
            order = order_for(g_bat, SW_STEPS)
            P, optim = restore(ck)
            rec_m, rec_p = [], []
            for it in range(SW_STEPS):
                step(P, optim, X, Y, order[it], act)
                if it + 1 in SW_SAVE:
                    with torch.no_grad():
                        W1, b1 = P[0].detach().double(), P[1].detach().double()
                        rec_m.append((W1 @ xbar + b1).numpy()); rec_p.append(((X.double() @ W1.T + b1) > 0).double().mean(0).numpy())
            res[f"{name}_d{j}_m"] = np.stack(rec_m); res[f"{name}_d{j}_p"] = np.stack(rec_p); res[f"{name}_d{j}_y"] = y.numpy()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"switch_{act}_s{seed}.npz", save_steps=np.array(SW_SAVE), **res)
    print(f"{act} s{seed}: switch done ({len(res) // 3} runs)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["replay", "switch"], required=True)
    ap.add_argument("--act", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckdir", default="results/reload_dial_1020/raw/ckpt")
    ap.add_argument("--out", default="results/reload_dial_1020/raw")
    ap.add_argument("--draws", type=int, default=5)
    a = ap.parse_args()
    t0 = time.monotonic()
    if a.stage == "replay": replay(a.act, a.seed, Path(a.ckdir))
    else: switch(a.act, a.seed, Path(a.ckdir), Path(a.out) / "switch", a.draws)
    print(f"elapsed {time.monotonic() - t0:.1f}s", flush=True)

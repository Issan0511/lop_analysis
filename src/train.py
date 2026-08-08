"""グループ ((exp, width) 単位) のベクトル化オンライン学習ループ。

各系列は batch=1 の plain SGD ([D][J])。系列 (条件×シード) を R 次元に平坦化して並列化
(仕様書 §8)。中心化 (enc=centered) は学習器入力の前処理であり、教師は生入力を見る。
"""
import os
import time
import csv
import torch

from .envs import SCREnv, LTUTarget, GaussEnv, MLPTeacher
from .nets import VecMLP
from .lop_metrics import compute_lop_metrics

SEED_BASE = {"A": 10000, "B": 20000}


def make_gens(exp, width, device, offset=0):
    """入力・教師・初期化・eval で generator を分離 (仕様書 §8)。"""
    base = SEED_BASE[exp] + width + offset
    gens = {}
    for i, name in enumerate(["init", "input", "teacher", "eval"]):
        g = torch.Generator(device=device)
        g.manual_seed(base + 100 * (i + 1))
        gens[name] = g
    return gens


def setup_group(gkey, runs, cfg, device):
    exp, width = gkey
    R = len(runs)
    gens = make_gens(exp, width, device)
    A, B = cfg["condA"], cfg["condB"]

    period = torch.tensor([r["period"] for r in runs], dtype=torch.long)  # CPU
    lr = torch.tensor([r["lr"] for r in runs], device=device)
    centered = torch.tensor([r["enc"] == "centered" for r in runs], device=device)

    if exp == "A":
        d = A["m"]
        env = SCREnv(R, A["m"], A["f"], period, gens["input"], device)
        teacher = LTUTarget(R, A["m"], A["target_hidden"], A["beta"], gens["teacher"], device)
    else:
        d = B["d"]
        cvals = [r["c"] for r in runs]
        env = GaussEnv(R, d, cvals, gens["input"], device)
        teacher = MLPTeacher(R, width, d, period, gens["teacher"], device)

    net = VecMLP(R, width, d, gens["init"], device)
    running_mean = torch.zeros(R, d, device=device)

    # LoP 計測用固定バッチ素材 (eval generator)
    N = cfg["common"]["eval_batch"]
    if exp == "A":
        eval_fixed = torch.randint(0, 2, (N, A["m"] - A["f"]),
                                   generator=gens["eval"], device=device).float()
    else:
        eval_fixed = torch.randn(N, d, generator=gens["eval"], device=device)

    return dict(exp=exp, width=width, R=R, d=d, env=env, teacher=teacher, net=net,
                running_mean=running_mean, lr=lr, centered=centered, period=period,
                eval_fixed=eval_fixed, runs=runs, device=device,
                alpha=A["center_alpha"])


def eval_batch(st):
    """現在の環境状態での計測用バッチ (x_raw [N,R,d], y [N,R])。"""
    if st["exp"] == "A":
        N = st["eval_fixed"].shape[0]
        f = st["env"].f
        flip = st["env"].flip_state[None].expand(N, -1, -1)              # [N,R,f]
        rnd = st["eval_fixed"][:, None, :].expand(-1, st["R"], -1)        # [N,R,m-f]
        x = torch.cat([flip, rnd], dim=2)
    else:
        x = st["env"].mu[None] + st["eval_fixed"][:, None, :]
    y = st["teacher"](x)
    return x, y


def save_ckpt(st, step, outdir):
    path = os.path.join(outdir, "ckpts", f"{st['exp']}_w{st['width']}_step{step}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(step=step,
                    net=st["net"].state_dict(),
                    env=st["env"].state_dict(),
                    teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(),
                    runs=st["runs"]), path)


def train_group(gkey, runs, cfg, device, outdir, total_steps=None, ckpts=None):
    C = cfg["common"]
    total = total_steps or C["total_steps"]
    ckpt_set = set(ckpts if ckpts is not None else C["checkpoints"])
    st = setup_group(gkey, runs, cfg, device)
    net, env, teacher = st["net"], st["env"], st["teacher"]
    alpha, centered, lr = st["alpha"], st["centered"], st["lr"]
    cmask = centered[:, None].float()

    loss_rows, lop_rows = [], []
    loss_acc = torch.zeros(st["R"], device=device)
    t0 = time.time()

    for t in range(total):
        if t in ckpt_set:
            save_ckpt(st, t, outdir)

        if st["exp"] == "B":
            teacher.t = env.t
            teacher.maybe_resample()
        x_raw = env.step()                                   # [R,d]
        y = teacher(x_raw)                                   # [R]

        x_in = x_raw - cmask * st["running_mean"]
        st["running_mean"].mul_(1 - alpha).add_(alpha * x_raw)

        pre, a, yhat = net.forward(x_in)
        delta = yhat - y
        gW, gb, gv, gc = net.grads(x_in, pre, a, delta)
        net.sgd_step(lr, gW, gb, gv, gc)

        loss_acc += delta ** 2
        if (t + 1) % C["loss_bin"] == 0:
            loss_rows.append((t + 1, (loss_acc / C["loss_bin"]).cpu().numpy().copy()))
            loss_acc.zero_()

        if (t + 1) % C["lop_every"] == 0 or t == 0:
            x_ev, y_ev = eval_batch(st)
            x_ev_in = x_ev - cmask[None] * st["running_mean"][None]
            m = compute_lop_metrics(net, x_ev_in, y_ev, cfg)
            lop_rows.append((t + 1 if t > 0 else 0, {k: v.cpu().numpy().copy() for k, v in m.items()}))

    if total in ckpt_set:
        save_ckpt(st, total, outdir)

    elapsed = time.time() - t0
    write_logs(st, loss_rows, lop_rows, outdir)
    return st, elapsed


def write_logs(st, loss_rows, lop_rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    gname = f"{st['exp']}_w{st['width']}"
    ids = [r["run_id"] for r in st["runs"]]

    with open(os.path.join(outdir, f"online_loss_{gname}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "run_id", "loss"])
        for step, arr in loss_rows:
            for i, rid in enumerate(ids):
                w.writerow([step, rid, f"{arr[i]:.6g}"])

    if lop_rows:
        keys = list(lop_rows[0][1].keys())
        with open(os.path.join(outdir, f"lop_metrics_{gname}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "run_id"] + keys)
            for step, m in lop_rows:
                for i, rid in enumerate(ids):
                    w.writerow([step, rid] + [f"{m[k][i]:.6g}" for k in keys])

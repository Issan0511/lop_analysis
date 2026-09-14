#!/usr/bin/env python3
"""pmnist_lopcmp_0905 — Permuted MNIST × 既存 LoP 手法（**未登録・事後**）。

    python3 src/pmnist_lopcmp_0905.py --stage checks
    python3 src/pmnist_lopcmp_0905.py --stage run --cells R:none,R:cbp:1e-4 --seeds 0,1,2

spec: 可塑性喪失/spec/PermutedMNIST_既存手法比較_spec_0905.md（走らせる前に §3 へ予測を登録済み）

宿主 `pmnist_0905` のデータ・置換・init・評価・検査をそのまま使い、**介入を活性化と直交な
軸として足すだけ**。`pmnist_0905` は 1 行も変えない。介入 `none` は恒等で、宿主の出力と
bit 一致することを S-null が検査する。

**実装は本 spec §5 の擬似コードであって原論文の再現ではない。** 結果を書くときは
「本 spec の CBP 実装に対して」と書く（§7 引用制限）。
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H          # 宿主。触らない

EXPERIMENT = "pmnist_lopcmp_0905"
CBP_DECAY = 0.99          # spec §5: 効用の移動平均の減衰
CBP_MATURITY = 100        # spec §5: 成熟閾値 m [step]


# ---------------------------------------------------------------- 介入
@dataclass
class Intervention:
    kind: str = "none"
    lam: float = 0.0          # l2 / l2init
    shrink: float = 1.0       # snp
    sigma: float = 0.0        # snp
    rho: float = 0.0          # cbp

    @property
    def tag(self) -> str:
        if self.kind == "none": return "none"
        if self.kind in ("l2", "l2init"): return f"{self.kind}:{self.lam:g}"
        if self.kind == "snp": return f"snp:{self.shrink:g}:{self.sigma:g}"
        if self.kind == "cbp": return f"cbp:{self.rho:g}"
        raise ValueError(self.kind)

    @staticmethod
    def parse(s: str) -> "Intervention":
        p = s.split(":")
        if p[0] == "none": return Intervention("none")
        if p[0] in ("l2", "l2init"): return Intervention(p[0], lam=float(p[1]))
        if p[0] == "snp": return Intervention("snp", shrink=float(p[1]), sigma=float(p[2]))
        if p[0] == "cbp": return Intervention("cbp", rho=float(p[1]))
        raise ValueError(s)


class CbpState:
    """spec §5 の CBP。隠れ層 1 と 2（DIMS[1], DIMS[2]）だけを対象にする。"""
    def __init__(self, seed: int, device):
        self.dev = device
        self.n = [H.DIMS[1], H.DIMS[2]]
        self.util = [torch.zeros(n, device=device, dtype=torch.float64) for n in self.n]
        self.age = [torch.zeros(n, device=device, dtype=torch.long) for n in self.n]
        self.debt = [0.0, 0.0]
        self.replaced = [0, 0]
        # 置換用の乱数は init と**別の系列**（init のストリームを消費しない）
        self.g = H.stream("cbp", seed)

    @torch.no_grad()
    def step(self, params, acts, rho: float):
        # acts = [a1, a2]（バッチ平均の |h|）、出力側は W2 (層1) と W3 (層2)
        W_out = [params[2], params[4]]
        for li in (0, 1):
            h = acts[li].abs().mean(dim=0).double()            # (n,)
            wsum = W_out[li].abs().sum(dim=0).double()         # (n,)
            self.util[li].mul_(CBP_DECAY).add_((1 - CBP_DECAY) * h * wsum)
            self.age[li] += 1
            self.debt[li] += rho * self.n[li]
            k = int(self.debt[li])
            if k <= 0: continue
            mature = self.age[li] > CBP_MATURITY
            if int(mature.sum()) == 0: continue
            u = self.util[li].clone()
            u[~mature] = float("inf")
            k = min(k, int(mature.sum()))
            idx = torch.topk(u, k, largest=False).indices
            self.debt[li] -= k
            self.replaced[li] += k
            # 入力側を引き直し、出力側を 0 に
            Win, bin_ = params[2 * li], params[2 * li + 1]
            fan_in = H.DIMS[li]
            bound = 1.0 / math.sqrt(fan_in)
            newW = ((torch.rand((k, fan_in), generator=self.g, dtype=torch.float32) * 2 - 1)
                    * bound).to(self.dev)
            newb = ((torch.rand((k,), generator=self.g, dtype=torch.float32) * 2 - 1)
                    * bound).to(self.dev)
            Win[idx] = newW
            bin_[idx] = newb
            W_out[li][:, idx] = 0.0
            self.age[li][idx] = 0
            self.util[li][idx] = 0.0


@torch.no_grad()
def snp_apply(params, p0_shapes, shrink: float, sigma: float, g):
    for p, (shape, bound) in zip(params, p0_shapes):
        eps = ((torch.rand(shape, generator=g, dtype=torch.float32) * 2 - 1) * bound).to(p.device)
        p.mul_(shrink).add_(sigma * eps)


def run_one(arm: str, seed: int, lr: float, n_tasks: int, iv: Intervention,
            mnist, device):
    """宿主 H.run_one の写し。差は (a) 介入フック 3 か所 (b) 返り値に置換数。"""
    act = H.ARMS[arm]
    params = H.init_params(seed, device)
    p0 = [p.detach().clone() for p in params] if iv.kind == "l2init" else None
    shapes = [(tuple(p.shape), 1.0 / math.sqrt(H.DIMS[i // 2])) for i, p in enumerate(params)]
    g_perm, g_data, g_batch = (H.stream("perm", seed), H.stream("data", seed),
                               H.stream("batch", seed))
    g_snp = H.stream("snp", seed)
    cbp = CbpState(seed, device) if iv.kind == "cbp" else None
    rows, diverged = [], {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm}

    for t in range(1, n_tasks + 1):
        perm = torch.randperm(784, generator=g_perm).to(device)
        idx = H.stratified_draw(mnist, g_data).to(device)
        order = torch.randperm(H.TASK_EXAMPLES, generator=g_batch).to(device)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        bad_step = torch.full((), -1, dtype=torch.long, device=device)
        for s in range(H.STEPS_PER_TASK):
            xb = xs[s * H.BATCH:(s + 1) * H.BATCH]
            yb = ys[s * H.BATCH:(s + 1) * H.BATCH]
            out = H.forward(params, xb, act)
            loss = torch.nn.functional.cross_entropy(out[4], yb)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                bad = ~torch.isfinite(loss)
                bad_step = torch.where((bad_step < 0) & bad,
                                       torch.tensor(s, device=device), bad_step)
                for i, (p, gr) in enumerate(zip(params, grads)):
                    if iv.kind == "l2":
                        gr = gr + 2.0 * iv.lam * p
                    elif iv.kind == "l2init":
                        gr = gr + 2.0 * iv.lam * (p - p0[i])
                    p -= lr * gr
                if cbp is not None:
                    cbp.step(params, [out[1], out[3]], iv.rho)
        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t,
                            step=(t - 1) * H.STEPS_PER_TASK + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "lr": lr, "iv": iv.tag, "task": t,
                         "acc": float("nan")})
            break
        m = H.evaluate(params, mnist, perm, act)
        rows.append({"arm": arm, "seed": seed, "lr": lr, "iv": iv.tag, "task": t, **m})
        if iv.kind == "snp":
            with torch.no_grad():
                snp_apply(params, shapes, iv.shrink, iv.sigma, g_snp)
    extra = dict(replaced=(cbp.replaced if cbp else [0, 0]))
    return rows, diverged, extra


# ---------------------------------------------------------------- 検査
def checks(device):
    out = {}
    mnist = H.Mnist(device)
    # S-null: none が宿主と bit 一致
    a, _, _ = run_one("R", 0, 0.1, 3, Intervention("none"), mnist, device)
    b, _ = H.run_one("R", 0, 0.1, 3, mnist, device)
    keys = [k for k in a[0] if k not in ("iv",)]
    same = all(all(_eq(x[k], y[k]) for k in keys) for x, y in zip(a, b))
    out["S_null"] = dict(pass_=bool(same), n_task=3)
    # S-l2-zero: lam=0 が none と bit 一致
    c, _, _ = run_one("R", 0, 0.1, 3, Intervention("l2", lam=0.0), mnist, device)
    out["S_l2_zero"] = dict(pass_=bool(all(all(_eq(x[k], y[k]) for k in keys)
                                           for x, y in zip(a, c))))
    d, _, _ = run_one("R", 0, 0.1, 3, Intervention("l2init", lam=0.0), mnist, device)
    out["S_l2init_zero"] = dict(pass_=bool(all(all(_eq(x[k], y[k]) for k in keys)
                                               for x, y in zip(a, d))))
    # S-snp-identity: shrink=1, sigma=0 が none と bit 一致
    e, _, _ = run_one("R", 0, 0.1, 3, Intervention("snp", shrink=1.0, sigma=0.0), mnist, device)
    out["S_snp_identity"] = dict(pass_=bool(all(all(_eq(x[k], y[k]) for k in keys)
                                                for x, y in zip(a, e))))
    # S-cbp-rate: 置換回数が rho*n*steps と ±1
    rho = 1e-3
    _, _, ex = run_one("R", 0, 0.1, 2, Intervention("cbp", rho=rho), mnist, device)
    steps = 2 * H.STEPS_PER_TASK
    exp1 = rho * H.DIMS[1] * steps - CBP_MATURITY * rho * H.DIMS[1]
    out["S_cbp_rate"] = dict(replaced=ex["replaced"], steps=steps,
                             expected_upper=rho * H.DIMS[1] * steps,
                             pass_=bool(ex["replaced"][0] <= rho * H.DIMS[1] * steps + 1
                                        and ex["replaced"][0] > 0))
    # S-cbp-zero: rho=0 が none と bit 一致
    f, _, _ = run_one("R", 0, 0.1, 3, Intervention("cbp", rho=0.0), mnist, device)
    out["S_cbp_zero"] = dict(pass_=bool(all(all(_eq(x[k], y[k]) for k in keys)
                                            for x, y in zip(a, f))))
    out["all_pass"] = all(v.get("pass_", True) for v in out.values() if isinstance(v, dict))
    return out


def _eq(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or x == y
    return x == y


def main():
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--stage", default="run", choices=["run", "checks"])
    ap.add_argument("--cells", default="", help="arm:iv[,...]  例 R:none,R:cbp:1e-4")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--out", default="results/pmnist_lopcmp_0905")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = H.setup(a.device)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    if a.stage == "checks":
        r = checks(device)
        (outdir / "checks.json").write_text(json.dumps(r, indent=2, default=str))
        print(json.dumps(r, indent=2, default=str)); return
    mnist = H.Mnist(device)
    seeds = [int(v) for v in a.seeds.split(",")]
    cells = []
    for c in a.cells.split(","):
        c = c.strip()
        if not c: continue
        arm, iv = c.split(":", 1)
        cells.append((arm, Intervention.parse(iv)))
    rows, divs, extras = [], [], []
    t0 = time.time()
    for arm, iv in cells:
        for sd in seeds:
            r, dv, ex = run_one(arm, sd, a.lr, a.tasks, iv, mnist, device)
            rows += r; divs.append(dv); extras.append(dict(arm=arm, iv=iv.tag, seed=sd, **ex))
            print(f"[{EXPERIMENT}] {arm}:{iv.tag} seed{sd} lr{a.lr} "
                  f"-> {'DIV' if dv['diverged'] else 'ok'} acc_final={r[-1].get('acc')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    H.write_csv(outdir / "per_task.csv", rows)
    (outdir / "provenance.json").write_text(json.dumps(dict(
        experiment=EXPERIMENT, unregistered=True,
        spec="可塑性喪失/spec/PermutedMNIST_既存手法比較_spec_0905.md",
        note="実装は spec §5 の擬似コードであって原論文の再現ではない",
        cells=[f"{x}:{y.tag}" for x, y in cells], seeds=seeds, lr=a.lr, tasks=a.tasks,
        device=str(device), git_hash=H.git_hash(), elapsed_sec=time.time() - t0,
        divergences=[d for d in divs if d["diverged"]], cbp_replacements=extras,
    ), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[{EXPERIMENT}] done in {time.time()-t0:.0f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main()

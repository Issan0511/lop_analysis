#!/usr/bin/env python3
"""追補 8: 2 ブロックの C3-2 で、切替の 100 更新だけ上流の共通成分を固定して L1 の変位を測る（事後・未登録）。
課題 t−1 の末（重み＋Adam）から課題 t の 100 更新を同じデータ列で再生する。腕:
  control  そのまま（元の走の損失と保存済みの更新 100 の重みにビット一致するはず）
  ffn0     FFN0 の出力 f₀ を f₀ − mean(f₀) + c₀ に（mean はバッチの全トークン、c₀ は窓の始まりの課題 t の probe のトークン平均）
  attn1    注意 1 の出力（係数を掛ける前）を同じ形で
L1 の Δμ/σ を 自分 Δw·ȳ_a・上流 w_a·Δȳ・交差 Δw·Δȳ に割り、unit ごとに保存する（両端とも課題 t の鎖の probe、σ は始まりの幅）。
  CHARGPT_NLAYER=2 python analysis/chargpt_cl_1019_quicklook/carrier_fix_m2.py --seed 0 --out results/chargpt_cl_1019/m2_carrier_fix
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
TASKS = (2, 3, 5, 10, 41, 45, 50)
ARMS = ("control", "ffn0", "attn1")


@torch.no_grad()
def l1_stats(model, x):
    cap = []; model.body(x, cap)
    y = model.h[1].n2(cap[2]).reshape(-1, C.D).double(); z = cap[3].reshape(-1, C.FF).double()
    return y.mean(0), z.std(0), model.h[1].mlp.up.weight.detach().double().clone()


def fix_common(mod, c):
    orig = mod.forward

    def fwd(*args, orig=orig, c=c, **kw):
        o = orig(*args, **kw)
        return o - o.mean((0, 1), keepdim=True) + c
    mod.forward = fwd


def run_one(arm, seed, t, src, cpu, ref_loss, dev):
    model = C.build_model("C3", "M", seed, dev)
    model.load_state_dict(torch.load(src / "ckpts" / f"task{t - 1}.pt", map_location=dev))
    probe = cpu["probe"].to(dev)
    target = {"ffn0": model.h[0].mlp, "attn1": model.h[1].attn}.get(arm)
    c = None
    if target is not None:
        got = {}
        hk = target.register_forward_hook(lambda m, i, o: got.__setitem__("o", o.detach()))
        with torch.no_grad(): model.body(probe)
        hk.remove(); c = got["o"].mean((0, 1), keepdim=True).clone(); fix_common(target, c)
    ya, sa, Wa = l1_stats(model, probe)
    eng = C.Engine(model, "C3", "M", graph=True)
    sd = torch.load(src / "ckpts" / f"task{t - 1}_adam.pt", map_location=dev)
    params = list(model.parameters())
    with torch.no_grad():
        for i, p in enumerate(params):
            st, sv = eng.opt.state[p], sd["state"][i]
            for k in ("exp_avg", "exp_avg_sq", "step"): st[k].copy_(sv[k].to(st[k].dtype))
    n = 10000; ctx = C.TaskCtx("M", seed, t, n, dev, alpha=0.01, W=None, cpu=cpu)
    bl = torch.zeros(100, device=dev); ba = torch.zeros(100, device=dev); lr_now = None
    for u in range(100):
        lr = eng.base_lr * min(1.0, ((t - 1) * n + u + 1) / C.WARMUP)
        if lr != lr_now: eng.set_lr(lr); lr_now = lr
        x, y = ctx.batch(u); eng.step(x, y, bl, ba, u)
    loss = bl.cpu().numpy()
    yb, _, Wb = l1_stats(model, probe)
    dW, dy = Wb - Wa, yb - ya
    own, up, cross = (dW @ ya) / sa, (Wa @ dy) / sa, (dW @ dy) / sa
    total = (Wb @ yb - Wa @ ya) / sa
    out = dict(own=own.cpu().numpy(), up=up.cpu().numpy(), cross=cross.cpu().numpy(), total=total.cpu().numpy(),
               resid=float((own + up + cross - total).abs().max()), loss_last10=float(loss[90:].mean()), loss=loss)
    if c is not None:                                      # 検査: 窓の末の probe 上の共通成分が c と等しい
        got = {}; hk = target.register_forward_hook(lambda m, i, o: got.__setitem__("o", o.detach()))
        with torch.no_grad(): model.body(probe)
        hk.remove(); out["fix_err"] = float((got["o"].mean((0, 1), keepdim=True) - c).abs().max() / c.abs().max())
    if arm == "control":
        out["loss_bitwise"] = bool(np.array_equal(loss, ref_loss))
        u100 = torch.load(src / "ckpts" / f"u100_task{t}.pt", map_location=dev)
        out["u100_bitwise"] = all(torch.equal(model.state_dict()[k], u100[k]) for k in u100)
    del eng, model, ctx; C._ENGINES.clear(); torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); assert C.L == 2
    C.setup_torch(); dev = torch.device("cuda"); src = ROOT / "results/chargpt_cl_1019/M2/C3" / f"seed{a.seed}"
    out = Path(a.out) / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    ref = np.load(src / "online.npz")["loss"]; gh, dirty = C.git_info(); t0 = time.time(); arrays, meta = {}, {}
    for t in TASKS:
        cpu = C.task_cpu_M(a.seed, t, 10000, 0.01)
        for arm in ARMS:
            r = run_one(arm, a.seed, t, src, cpu, ref[t - 1][:100], dev)
            for k in ("own", "up", "cross", "total", "loss"): arrays[f"{arm}_t{t}_{k}"] = r[k]
            meta[f"{arm}_t{t}"] = {k: v for k, v in r.items() if k not in ("own", "up", "cross", "total", "loss")}
            print(f"[seed {a.seed}] t{t} {arm}: Δμ/σ 平均 {r['total'].mean():+.3f}（上流＋交差 {(r['up'] + r['cross']).mean():+.3f}）"
                  f" 損失末 {r['loss_last10']:.3f} {({k: v for k, v in meta[f'{arm}_t{t}'].items() if k not in ('loss_last10',)})}", flush=True)
    np.savez_compressed(out / "units.npz", **arrays)
    (out / "meta.json").write_text(json.dumps(dict(seed=a.seed, tasks=TASKS, arms=ARMS, runs=meta, git_hash=gh, dirty_src=dirty, gpu=torch.cuda.get_device_name(0),
                                                   seconds=time.time() - t0, grade="事後・未登録（追補 8）"), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

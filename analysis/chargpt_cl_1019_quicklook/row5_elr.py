#!/usr/bin/env python3
"""spec 追補 9.C: 実効学習率だけを保つ腕 ELR と、文字どおりの形の診断（事後の介入）。
  ELR:     受け手 C3-2、課題 1 末（重み＋Adam）から課題 2〜100。各 Adam の更新のあと L0 の W_up の各行を課題 1 末のノルムに射影し直す。
           L0 の前活性に学習しない倍率 k_i(t) = ‖w_i^{C3-2}(t)‖ / ‖w_i^{C3-2}(1)‖ を掛ける（同じ seed の C3-2 の unit_stats の rownorm（各課題末）を、
           課題の中で更新ごとに線形補間）。L1 は素のまま。
  literal: seed 0、課題 1 末から課題 2〜20。L0 の W_up の各行の更新量を ‖w_i‖/‖w_i,1‖ 倍する（lr を unit ごとに ‖w‖/‖w₁‖ 倍と同じ）。非有限で止める。
  CHARGPT_NLAYER=2 python analysis/chargpt_cl_1019_quicklook/row5_elr.py --mode elr --seed 0 --out <dir>
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
M2 = ROOT / "results/chargpt_cl_1019/M2"
HOOKS = dict(pre=[], post=[])
_prev = C.Engine._step_body


def _body(s):
    with torch.no_grad():
        for f in HOOKS["pre"]: f()
    out = _prev(s)
    with torch.no_grad():
        for f in HOOKS["post"]: f()
    return out


C.Engine._step_body = _body


def three_phase(loss, H):
    ex = loss.astype(np.float64) - H; ma = np.convolve(ex, np.ones(50) / 50, "valid"); hit = np.nonzero(ma <= 0.15)[0]
    return dict(T015=(int(hit[0]) + 50 if len(hit) else 10000), reached=bool(len(hit)), E1000=float(ex[950:1050].mean()), ELast=float(ex[-1000:].mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["elr", "literal"], required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--to_task", type=int, default=None); ap.add_argument("--n_per_task", type=int, default=10000); ap.add_argument("--out", required=True)
    a = ap.parse_args(); assert C.L == 2
    C.setup_torch(); dev = torch.device("cuda"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0w = time.time()
    seed = a.seed; to_task = a.to_task or (100 if a.mode == "elr" else 20); n = a.n_per_task
    src = M2 / "C3" / f"seed{seed}"
    model = C.build_model("C3", "M", seed, dev); model.load_state_dict(torch.load(src / "ckpts" / "task1.pt", map_location=dev))
    mlp0 = model.h[0].mlp; W = mlp0.up.weight
    n1 = W.detach().norm(dim=1).clone()                                       # 課題 1 末の行ノルム
    info = dict(mode=a.mode, seed=seed, from_task=1, to_task=to_task)
    k = torch.ones(C.FF, device=dev)
    if a.mode == "elr":
        rn = torch.from_numpy(np.load(src / "unit_stats.npz")["rownorm"][:, 0, :]).to(dev).float()   # [課題−1, unit]
        assert torch.allclose(rn[0], n1, rtol=1e-4), "unit_stats の rownorm（課題 1 末）が checkpoint と合わない"
        ktab = rn / rn[0:1]                                                    # k_i(t) at task-end t（行 t−1）
        mlp0.phi = lambda z, k=k: F.relu(k * z)
        HOOKS["post"].append(lambda W=W, n1=n1: W.mul_((n1 / W.norm(dim=1).clamp_min(1e-30))[:, None]))
    else:
        W_old = torch.zeros_like(W)
        HOOKS["pre"].append(lambda W=W, W_old=W_old: W_old.copy_(W))
        HOOKS["post"].append(lambda W=W, W_old=W_old, n1=n1: W.copy_(W_old + (W - W_old) * (W_old.norm(dim=1) / n1)[:, None]))
    eng = C.Engine(model, "C3", "M", graph=True)
    sd = torch.load(src / "ckpts" / "task1_adam.pt", map_location=dev); params = list(model.parameters()); assert len(sd["state"]) == len(params)
    with torch.no_grad():
        for i, p in enumerate(params):
            for kk in ("exp_avg", "exp_avg_sq", "step"): eng.opt.state[p][kk].copy_(sd["state"][i][kk].to(eng.opt.state[p][kk].dtype))
    probe_u = C.uniform_probe_M(seed, dev)
    tasks = list(range(2, to_task + 1)); online = np.zeros((len(tasks), n), np.float32); rows, layer_rows = [], []
    for ti, t in enumerate(tasks):
        cpu = C.task_cpu_M(seed, t, n, 0.01); ctx = C.TaskCtx("M", seed, t, n, dev, alpha=0.01, W=None, cpu=cpu); tt = time.time()
        Wprev = W.detach().clone(); bl = torch.zeros(n, device=dev); ba = torch.zeros(n, device=dev); eng.set_lr(eng.base_lr)
        for u in range(n):
            if a.mode == "elr":
                k.copy_(ktab[t - 2] + (ktab[t - 1] - ktab[t - 2]) * ((u + 1) / n))        # 課題 t−1 末 → 課題 t 末を更新ごとに線形補間
            x, y = ctx.batch(u); eng.step(x, y, bl, ba, u)
            if a.mode == "literal" and (u + 1) % 100 == 0 and not (torch.isfinite(bl[u]) and torch.isfinite(W).all()):
                bl[u + 1:] = float("nan"); break                                # 非有限になった 100 更新の区切りで止める
        loss = bl.cpu().numpy(); online[ti] = loss
        for pn, xp in dict(chain=ctx.probe, uniform=probe_u).items():
            for li, r in enumerate(C.measure(model, xp)): layer_rows.append(dict(task=t, probe=pn, layer=li, **r["agg"]))
        with torch.no_grad():
            got = {}; hk = mlp0.register_forward_hook(lambda m, i, o: got.__setitem__("f0", o.detach())); model.body(ctx.probe); hk.remove()
            cosv = (F.normalize(W, dim=1) * F.normalize(Wprev, dim=1)).sum(1)
        fin = bool(np.isfinite(loss).all()) and bool(torch.isfinite(W).all())
        row = dict(task=t, H=ctx.H, **C.summarize_online("M", loss, loss, ctx.H), **(three_phase(loss, ctx.H) if fin else {}),
                   w0_norm_med=float(W.norm(dim=1).median()), w0_norm_over_n1_med=float((W.norm(dim=1) / n1).median()), rot0_med=float((1 - cosv).median()),
                   ffn0_out_rms=float(got["f0"].double().pow(2).mean().sqrt()), k_med=float(k.median()), finite=fin)
        rows.append(row)
        L0 = [r for r in layer_rows if r["task"] == t and r["probe"] == "chain" and r["layer"] == 0][0]
        print(f"[{a.mode} s{seed}] task {t} online {row['online_all']:.4f} | L0 死 {L0['frac_p_lt005']:.2f} | ‖w0‖ 中央 {row['w0_norm_med']:.3g}（×{row['w0_norm_over_n1_med']:.3g}） 回転 {row['rot0_med']:.2e}"
              f" FFN0 rms {row['ffn0_out_rms']:.2f} k {row['k_med']:.2f} | {time.time() - tt:.0f}s", flush=True)
        if t in (50, 100) and fin:
            torch.save(model.state_dict(), out / f"state_task{t}.pt")
        if not fin: info["stopped_nonfinite_at"] = t; break
        del ctx

    def wcsv(path, rws):
        keys = list(dict.fromkeys(k_ for r in rws for k_ in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rws: w.writerow({k_: (f"{v:.9g}" if isinstance(v, float) else v) for k_, v in r.items()})
    wcsv(out / "per_task.csv", rows); wcsv(out / "layer_stats.csv", layer_rows); np.savez_compressed(out / "online.npz", loss=online[:len(rows)])
    gh, dirty = C.git_info()
    (out / "provenance.json").write_text(json.dumps(dict(script="row5_elr.py", **info, n_per_task=n, git_hash=gh, dirty_src=dirty, gpu=torch.cuda.get_device_name(0),
                                                       host=__import__("socket").gethostname(), seconds=time.time() - t0w, grade="spec 追補 9.C"), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

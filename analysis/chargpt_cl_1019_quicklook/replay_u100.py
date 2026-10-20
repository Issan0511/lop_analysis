#!/usr/bin/env python3
"""課題 k の末（重み＋Adam）から課題 k+1 を 100 更新だけ GPU で再生し、更新 100 の重みと Adam の状態を保存する（ρ の分割の計量 D 用、事後）。
保存済みの u100_task{k+1}.pt があれば、再生した重みとビット一致するかを確かめる。
  python analysis/chargpt_cl_1019_quicklook/replay_u100.py --arm C3 --ckpts <dir> --ks 1 2 3 --out <dir>
"""
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--ckpts", required=True); ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--n_per_task", type=int, default=10000)
    a = ap.parse_args(); C.setup_torch(); dev = torch.device("cuda"); src, out = Path(a.ckpts), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = {}
    for k in a.ks:
        t0 = time.time()
        model = C.build_model(a.arm, "M", a.seed, dev); model.load_state_dict(torch.load(src / f"task{k}.pt", map_location=dev))
        eng = C.Engine(model, a.arm, "M", graph=True)
        sd = torch.load(src / f"task{k}_adam.pt", map_location=dev); params = list(model.parameters())
        with torch.no_grad():
            for i, p in enumerate(params):
                st, sv = eng.opt.state[p], sd["state"][i]
                for key in ("exp_avg", "exp_avg_sq", "step"): st[key].copy_(sv[key].to(st[key].dtype))
        t = k + 1; ctx = C.TaskCtx("M", a.seed, t, a.n_per_task, dev, alpha=0.01, W=None, cpu=C.task_cpu_M(a.seed, t, a.n_per_task, 0.01))
        buf_l = torch.zeros(100, device=dev); buf_a = torch.zeros(100, device=dev); lr_now = None
        for u in range(100):
            lr = eng.base_lr * min(1.0, ((t - 1) * a.n_per_task + u + 1) / C.WARMUP)
            if lr != lr_now: eng.set_lr(lr); lr_now = lr
            x, y = ctx.batch(u); eng.step(x, y, buf_l, buf_a, u)
        torch.save(model.state_dict(), out / f"u100_task{t}.pt"); torch.save(eng.opt.state_dict(), out / f"u100_task{t}_adam.pt")
        ref = src / f"u100_task{t}.pt"; match = None
        if ref.exists():
            r = torch.load(ref, map_location=dev); match = all(torch.equal(model.state_dict()[n], r[n]) for n in r)
        log[str(t)] = dict(bitwise_equal_to_saved_u100=match, seconds=round(time.time() - t0, 1))
        print(f"[{a.arm}] 課題 {k} の末 → 課題 {t} 更新 100: 保存済みとビット一致 {match}（{time.time() - t0:.0f}s）", flush=True)
        del model, eng, ctx; torch.cuda.empty_cache()
    (out / f"replay_u100_{a.arm}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()

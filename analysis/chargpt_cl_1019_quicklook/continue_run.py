#!/usr/bin/env python3
"""課題 T の末の checkpoint（重み＋Adam の状態）から課題 T+1〜T2 を続けて回す（下見・未登録）。
  python analysis/chargpt_cl_1019_quicklook/continue_run.py --arm SiLU --src results/.../M/SiLU/seed1000 --from_task 50 --to_task 100 --out results/.../M/SiLU_cont/seed1000
  --n_per_task 300 --to_task 26 で再開の検査（元の走の online.npz とビット一致するはず）。
記録は本走と同じ形（per_task.csv・layer_stats.csv・online.npz・unit_stats.npz・fresh.csv）だが、課題番号は from_task+1 から。fresh は to_task だけ。"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--from_task", type=int, required=True); ap.add_argument("--to_task", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--n_per_task", type=int, default=10000); ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--fresh", type=int, default=1)
    ap.add_argument("--save_tasks", type=int, nargs="*", default=[], help="この課題の末で重みと Adam の状態を保存する（to_task は常に保存）")
    a = ap.parse_args()
    C.setup_torch(); dev = torch.device(a.device); src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "ckpts").mkdir(exist_ok=True)
    gh, dirty = C.git_info(); t_start = time.time()
    model = C.build_model(a.arm, "M", a.seed, dev)
    model.load_state_dict(torch.load(src / "ckpts" / f"task{a.from_task}.pt", map_location=dev))
    eng = C.Engine(model, a.arm, "M", graph=(dev.type == "cuda"))
    # Adam の状態をグラフが指す tensor に書き込む（load_state_dict は tensor を差し替えるので使えない）
    sd = torch.load(src / "ckpts" / f"task{a.from_task}_adam.pt", map_location=dev)
    params = list(model.parameters()); assert len(sd["state"]) == len(params), (len(sd["state"]), len(params))
    with torch.no_grad():
        for i, p in enumerate(params):
            st, saved = eng.opt.state[p], sd["state"][i]
            for k in ("exp_avg", "exp_avg_sq", "step"):
                st[k].copy_(saved[k].to(st[k].dtype))
    probe_u = C.uniform_probe_M(a.seed, dev)
    rows, layer_rows, fresh_rows = [], [], []
    tasks = list(range(a.from_task + 1, a.to_task + 1)); n = a.n_per_task
    online = np.zeros((len(tasks), n), np.float32)
    unit_keys = ("mu", "sd", "q", "t_wgm", "t_wbeta", "t_b", "cos")
    U = {f"{p}_{k}": np.zeros((len(tasks), C.L, C.FF), np.float32) for p in ("chain", "uniform") for k in unit_keys}
    U.update({f"{p}_dead": np.zeros((len(tasks), C.L, C.FF), bool) for p in ("chain", "uniform")})
    init_sha = None
    for ti, t in enumerate(tasks):
        cpu = C.task_cpu_M(a.seed, t, n, a.alpha)
        ctx = C.TaskCtx("M", a.seed, t, n, dev, alpha=a.alpha, W=None, cpu=cpu)
        tt0 = time.time()
        loss, aux = C.run_task(model, eng, ctx, warmup_from=(t - 1) * n)
        online[ti] = loss
        for p, xp in dict(chain=ctx.probe, uniform=probe_u).items():
            for li, r in enumerate(C.measure(model, xp)):
                layer_rows.append(dict(task=t, u=n, probe=p, layer=li, **r["agg"]))
                for k in unit_keys: U[f"{p}_{k}"][ti, li] = r["units"][k]
                U[f"{p}_dead"][ti, li] = r["dead"]
        endm = C.eval_M(model, ctx.eval_seq, ctx.H)
        row = dict(task=t, n=n, H=ctx.H, **C.summarize_online("M", loss, aux, ctx.H), end_metric=endm, data_sha=ctx.data_sha[:16], rule_sha=ctx.rule_sha[:16]); rows.append(row)
        print(f"[M/{a.arm}/s{a.seed} cont] task {t}/{a.to_task} online_all {row['online_all']:.4f} end {endm:.4f} {time.time() - tt0:.0f}s", flush=True)
        if t in a.save_tasks and t != a.to_task:
            torch.save(model.state_dict(), out / "ckpts" / f"task{t}.pt"); torch.save(eng.opt.state_dict(), out / "ckpts" / f"task{t}_adam.pt")
        if t == a.to_task:
            torch.save(model.state_dict(), out / "ckpts" / f"task{t}.pt"); torch.save(eng.opt.state_dict(), out / "ckpts" / f"task{t}_adam.pt")
            if a.fresh:
                fm = C.build_model(a.arm, "M", a.seed, dev); fe = C.Engine(fm, a.arm, "M", graph=(dev.type == "cuda"))
                fl, fa = C.run_task(fm, fe, ctx, warmup_from=0); fend = C.eval_M(fm, ctx.eval_seq, ctx.H)
                fresh_rows.append(dict(task=t, n=n, H=ctx.H, **C.summarize_online("M", fl, fa, ctx.H), end_metric=fend, data_sha=ctx.data_sha[:16]))
                np.savez_compressed(out / f"fresh_online_t{t}.npz", loss=fl, aux=fa)
                print(f"[M/{a.arm}/s{a.seed} cont] fresh t{t} online_all {fresh_rows[-1]['online_all']:.4f}", flush=True)
        del ctx
    def wcsv(path, rws):
        if not rws: return
        keys = list(dict.fromkeys(k for r in rws for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rws: w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    wcsv(out / "per_task.csv", rows); wcsv(out / "layer_stats.csv", layer_rows); wcsv(out / "fresh.csv", fresh_rows)
    np.savez_compressed(out / "online.npz", loss=online); np.savez_compressed(out / "unit_stats.npz", tasks=np.array(tasks), **U)
    (out / "provenance.json").write_text(json.dumps(dict(script="continue_run.py", arm=a.arm, seed=a.seed, src=str(src), from_task=a.from_task, to_task=a.to_task, n_per_task=n, alpha=a.alpha,
                                                       git_hash=gh, dirty_src=dirty, device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", host=__import__("socket").gethostname(),
                                                       seconds=time.time() - t_start, grade="下見・未登録・続き走"), indent=1, ensure_ascii=False))
    print(f"[M/{a.arm}/s{a.seed} cont] done {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

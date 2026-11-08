#!/usr/bin/env python3
"""head0_cap_1108: 残った網の容量を同じ幅の新品と並べる（spec specs/spec_head0_mlp_1108.md 追補 A）。

本線 S を課題 1..tasks_from-1 まで engine と完全一致で再生し（検査 C1）、課題 tasks_from..tasks_to の各切替で
  - H0Ab: 本体の H0A と同じ出発状態（head = 0、Adam 全部新品）を別のミニバッチ順 batch_b|{t} で回す
  - fresh_k: 784-k-10 の新品（乱数列 head0_mlp_1108|cap_init|{a|b}|{k}|{t}、engine と同じ分布）を engine の順で回す
    k は格子 ∪ {k_start, k_end}（0 は除く。k = 0 の漸近は最頻クラスの割合 maj_t を算術で置く）
を回してから本線 S を進める。演算は src/head0_mlp_1108.py の run_arm / step（engine と同じ）をそのまま使う。
k_start = 切替時の live state で p⁺ ≥ 0.005 の unit 数、k_end = 本体の H0A の課題 t 末の p⁺ ≥ 0.005 の数（保存済み raw）。
"""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
import torch

from src.head0_mlp_1108 import (SAVED, WORKTREE, N, H, K, ENGINE_STEPS, EVAL_EVERY, stream, load_X, initial,
                                initial_from, order_for, step, snap, restore, h0stream, new_params, new_adam,
                                maxabs, Ctx, unit_stats, run_arm)

torch.set_num_threads(1)
PREV = Path("/home/issan/Projects/obsidian-research-data/head0_mlp_1108/results/head0_mlp_1108/raw")
ALIVE = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True, choices=["LR", "ELU", "R", "GELU", "SILU"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results/head0_mlp_1108/cap/raw")
    ap.add_argument("--tasks-from", type=int, default=21)
    ap.add_argument("--tasks-to", type=int, default=30)
    ap.add_argument("--grid", default="3,6,13,25,45,74,100")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.tasks_to, a.grid, a.out = a.tasks_from, "3,100", "results/_smoke/head0_cap_1108/raw"
    out = Path(a.out)
    if not out.is_absolute(): out = WORKTREE / out
    act, seed, t_from, t_to = a.act, a.seed, a.tasks_from, a.tasks_to
    grid = sorted({int(s) for s in a.grid.split(",") if s.strip()})
    assert 2 <= t_from <= t_to
    t_start = time.monotonic()

    X, subset = load_X(seed)
    saved = np.load(SAVED / f"{act}_K10_s{seed}.npz")
    assert np.array_equal(saved["subset"], subset), "subset が保存済みと違う"
    n_saved = saved["y"].shape[0]
    assert t_to <= n_saved
    prev = np.load(PREV / f"{act}_s{seed}.npz")
    ctx = Ctx(X, act)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P = new_params(initial(seed, (784, H, K)))
    optim = new_adam(P)
    init_maxabs = maxabs(P, [saved[f"init{q}"] for q in range(4)])

    check_end = np.full(t_to, np.nan)
    check_pp = []; k_start_l = []; k_end_l = []; tasks_run = []
    res = {}; per_task = []; label_check = []

    for t in range(1, t_to + 1):
        t_task = time.monotonic()
        y = torch.randint(K, (N,), generator=glabel)
        order = order_for(gbatch, ENGINE_STEPS)
        label_check.append(bool(np.array_equal(y.numpy(), saved["y"][t - 1])))
        Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
        info = dict(task=t)

        if t >= t_from:
            u_live, _ = unit_stats(P, ctx)
            pplus_start = u_live[1]
            k_start = int((pplus_start >= ALIVE).sum())
            pp_diff = float(np.abs(prev[f"S_t{t:02d}_unit"][0, 1] - pplus_start).max())
            k_end = int((prev[f"H0A_t{t:02d}_unit"][1, 1] >= ALIVE).sum())
            check_pp.append(pp_diff); k_start_l.append(k_start); k_end_l.append(k_end); tasks_run.append(t)
            res[f"y_t{t:02d}"] = y.numpy().astype(np.int64)
            res[f"maj_t{t:02d}"] = np.array(float(np.bincount(y.numpy(), minlength=K).max()) / N)
            ck = snap(P, optim, (t - 1) * ENGINE_STEPS, float("nan"))

            # H0Ab: head = 0、Adam 全部新品、別の順
            ck2 = dict(ck); ck2.update({f"step{q}": np.array(0.0) for q in range(4)})
            Pb, ob = restore(ck2)
            with torch.no_grad():
                Pb[2].data.zero_(); Pb[3].data.zero_()
            order_b = order_for(h0stream(f"batch_b|{t}", seed), ENGINE_STEPS)
            rec = run_arm(Pb, ob, ctx, y, Y, order_b, ENGINE_STEPS)[0]
            res[f"H0Ab_t{t:02d}_eval"] = rec["eval"]; res[f"H0Ab_t{t:02d}_unit"] = rec["unit"]
            info["H0Ab_end_acc"] = float(rec["eval"][-50:, 0].mean())
            info["H0A_backup_end_acc"] = float(prev[f"H0A_t{t:02d}_eval"][-50:, 0].mean())
            del ck, ck2, Pb, ob

            # fresh_k
            ks = sorted((set(grid) | {k_start, k_end}) - {0})
            res[f"k_list_t{t:02d}"] = np.array(ks, dtype=np.int64)
            fresh_end = {}
            for k in ks:
                for rep in ("a", "b"):
                    p0 = initial_from(h0stream(f"cap_init|{rep}|{k}|{t}", seed), dims=(784, k, K))
                    Pf = new_params(p0)
                    rec = run_arm(Pf, new_adam(Pf), ctx, y, Y, order, ENGINE_STEPS)[0]
                    key = f"fresh_k{k:03d}{rep}_t{t:02d}"
                    res[f"{key}_eval"] = rec["eval"]; res[f"{key}_unit"] = rec["unit"]; res[f"{key}_head"] = rec["head"]
                    fresh_end[f"{k}{rep}"] = round(float(rec["eval"][-50:, 0].mean()), 4)
            info.update(k_start=k_start, k_end=k_end, pplus_start_maxabs=pp_diff, k_list=ks, fresh_end_acc=fresh_end,
                        maj=float(res[f"maj_t{t:02d}"]))

        # 本線 S
        for it in range(ENGINE_STEPS):
            step(P, optim, X, Y, order[it], act)
        check_end[t - 1] = maxabs(P, [saved[f"end{q}"][t - 1] for q in range(4)])
        el = time.monotonic() - t_task
        info.update(elapsed_s=round(el, 2), end_maxabs=check_end[t - 1])
        per_task.append(info)
        msg = f"{act} s{seed} task {t:2d} end_maxabs {check_end[t-1]}"
        if t >= t_from:
            msg += (f" k_start {info['k_start']} k_end {info['k_end']} pp_diff {info['pplus_start_maxabs']}"
                    f" H0Ab {info['H0Ab_end_acc']:.3f} (H0A {info['H0A_backup_end_acc']:.3f}) fresh {info['fresh_end_acc']}")
        print(msg + f" ({el:.1f}s, total {time.monotonic()-t_start:.0f}s)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    res["k_start"] = np.array(k_start_l, dtype=np.int64); res["k_end"] = np.array(k_end_l, dtype=np.int64)
    res["tasks"] = np.array(tasks_run, dtype=np.int64)
    res["check_end_maxabs"] = check_end; res["check_pplus_start_maxabs"] = np.array(check_pp)
    np.savez_compressed(out / f"{act}_s{seed}.npz", **res)
    try:
        head = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        head = f"error: {e}"
    check = dict(init_maxabs=init_maxabs, labels_equal_saved=all(label_check),
                 check_end_maxabs=check_end.tolist(), check_pplus_start_maxabs=check_pp)
    meta = dict(act=act, seed=seed, tasks_from=t_from, tasks_to=t_to, grid=grid, alive_threshold=ALIVE,
                steps=ENGINE_STEPS, eval_every=EVAL_EVERY, smoke=a.smoke, torch=torch.__version__, git_head=head,
                prev_raw=str(PREV / f"{act}_s{seed}.npz"),
                notes=["k = 0 は走らない（maj_t が漸近）", "H0Ab の Adam は step=0 で全部新品（本体 H0A と同じ作り）",
                       "fresh_k の Adam は新品、順は engine の order"],
                elapsed_s=round(time.monotonic() - t_start, 1), per_task=per_task, check=check)
    (out / f"{act}_s{seed}.json").write_text(json.dumps(meta, indent=1))

    # 検査（出力を書いた後）
    assert init_maxabs == 0.0, f"engine の初期値と一致しない {init_maxabs}"
    assert all(label_check), "ラベルが保存済みと違う"
    assert np.all(check_end == 0.0), f"C1 end 不一致 {check_end}"
    assert all(v == 0.0 for v in check_pp), f"pplus_start が本体の S と不一致 {check_pp}"
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()

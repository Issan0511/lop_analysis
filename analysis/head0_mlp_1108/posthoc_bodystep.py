#!/usr/bin/env python3
"""head0_mlp_1108 事後記録: 課題 T=25 の切替後 4,000 更新で、体と head の 1 更新の変位・体の勾配・体の Adam 二次モーメントの尺度・訓練精度を
S / H0 / H0A の 3 腕で記録する。本線は src/head0_mlp_1108.py と同じ乱数順で課題 1..T−1 を回し、保存済み end*[T−2] と完全一致を確かめる。
数値だけを書き、読みは書かない。

実行: cd <worktree> && PYTHONPATH=. python -m analysis.head0_mlp_1108.posthoc_bodystep
      （--only ACT で 1 活性化だけ計算。既定は 4 つを並列の子プロセスで計算して表と図を書く）
"""
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

from src.head0_mlp_1108 import (load_X, stream, initial, order_for, new_params, new_adam, snap, restore, step,
                                 accuracy, activ, ENGINE_STEPS, K, N, H, SAVED, EVAL_EVERY, WORKTREE)

torch.set_num_threads(1)
T = 25
RUNS = [("LR", 0), ("ELU", 0), ("SILU", 0), ("GELU", 0)]
ARMS = ["S", "H0", "H0A"]
OUT = WORKTREE / "results/head0_mlp_1108/posthoc"
NAME = {"LR": "leaky", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}


def run_branch(P, optim, X, y, Y, order, act):
    steps = order.shape[0]
    rec = {k: np.zeros(steps) for k in ("body_step", "head_step", "body_grad", "v_body_rms")}
    acc = np.zeros(steps // EVAL_EVERY + 1)
    acc[0] = accuracy(P, X, y, act)
    for it in range(steps):
        idx = order[it]
        xb, yb = X[idx], Y[idx]
        z = xb @ P[0].T + P[1]
        a = activ(z, act); logits = a @ P[2].T + P[3]
        loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
        optim.zero_grad(set_to_none=True); loss.backward()
        rec["body_grad"][it] = float(sum(P[q].grad.double().pow(2).sum() for q in (0, 1)).sqrt())
        old = [P[q].detach().double().clone() for q in range(4)]
        optim.step()
        d = [float((P[q].detach().double() - old[q]).pow(2).sum()) for q in range(4)]
        rec["body_step"][it] = np.sqrt(d[0] + d[1]); rec["head_step"][it] = np.sqrt(d[2] + d[3])
        vs = [optim.state[P[q]]["exp_avg_sq"].double() for q in (0, 1) if P[q] in optim.state and "exp_avg_sq" in optim.state[P[q]]]
        if vs:
            tot = sum(float(v.sum()) for v in vs); n = sum(v.numel() for v in vs)
            rec["v_body_rms"][it] = np.sqrt(tot / n)
        if (it + 1) % EVAL_EVERY == 0:
            acc[(it + 1) // EVAL_EVERY] = accuracy(P, X, y, act)
    rec["acc"] = acc
    return rec


def compute(act, seed):
    t0 = time.monotonic()
    X, subset = load_X(seed)
    saved = np.load(SAVED / f"{act}_K10_s{seed}.npz")
    assert np.array_equal(saved["subset"], subset), "subset が保存済みと違う"
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P = new_params(initial(seed, (784, H, K))); optim = new_adam(P)
    for t in range(1, T):
        y = torch.randint(K, (N,), generator=glabel)
        order = order_for(gbatch, ENGINE_STEPS)
        assert np.array_equal(y.numpy(), saved["y"][t - 1]), f"課題 {t} のラベルが保存済みと違う"
        Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
        for it in range(ENGINE_STEPS):
            step(P, optim, X, Y, order[it], act)
    check = max(float(np.abs(P[q].detach().numpy() - saved[f"end{q}"][T - 2]).max()) for q in range(4))
    if check != 0.0:
        sys.exit(f"{act} s{seed}: 課題 {T-1} 末の θ が保存済み end*[{T-2}] と一致しない（max abs diff {check}）。中止")
    print(f"{act} s{seed}: replay to end of task {T-1} bit-exact ({time.monotonic()-t0:.0f}s)", flush=True)

    y = torch.randint(K, (N,), generator=glabel)
    order = order_for(gbatch, ENGINE_STEPS)
    if T <= saved["y"].shape[0]:
        assert np.array_equal(y.numpy(), saved["y"][T - 1]), f"課題 {T} のラベルが保存済みと違う"
    Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
    ck = snap(P, optim, (T - 1) * ENGINE_STEPS, float("nan"))
    out = {"y": y.numpy(), "check_end_maxabs": np.array(check), "T": np.array(T)}
    for arm in ARMS:
        if arm == "S":
            Pb, ob = restore(ck)
        elif arm == "H0":
            Pb, ob = restore(ck)
            with torch.no_grad(): Pb[2].zero_(); Pb[3].zero_()
            ob.state.pop(Pb[2], None); ob.state.pop(Pb[3], None)
        else:
            ck2 = dict(ck); ck2.update({f"step{q}": np.array(0.0) for q in range(4)})
            Pb, ob = restore(ck2)
            with torch.no_grad(): Pb[2].zero_(); Pb[3].zero_()
        rec = run_branch(Pb, ob, X, y, Y, order, act)
        for k, v in rec.items(): out[f"{arm}_{k}"] = v
        print(f"{act} s{seed} {arm}: acc end {rec['acc'][-1]:.3f} ({time.monotonic()-t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"bodystep_{act}_s{seed}_t{T}.npz", **out)


def fmt(x):
    if x == 0: return "0"
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e4 else f"{x:.2e}"


def report():
    W = [("1–100", 0, 100), ("101–1,000", 100, 1000), ("1,001–4,000", 1000, 4000)]
    L = ["# head0_mlp_1108 事後記録: 課題 25 の切替後の体の 1 更新の変位", "",
         "本線を課題 1〜24 まで回し、課題 24 末の θ が保存済みの値と完全に一致することを確かめた。",
         "課題 25 の頭から S・H0・H0A の 3 腕を同じミニバッチ順で 4,000 更新回した。",
         "seed はどれも 0 である。", "",
         "記号は次のとおり。",
         "- 体の変位: 1 更新での体（第 1 層の重みと bias）の変化のノルム ‖Δθ_body‖。",
         "- head の変位: 同じく head（第 2 層の重みと bias）の変化のノルム。",
         "- 体の勾配: 更新前の体の勾配のノルム。",
         "- v の尺度: 更新後の体の Adam 二次モーメントの平方根の RMS、√mean(v)。状態が無ければ 0。",
         "- 精度: 1,200 枚全部での訓練精度。", "",
         "H0 の更新 1 では head が 0 なので体の勾配は厳密に 0 である。",
         "そのため H0 の更新 1 の体の変位は、前の課題から残った一次モーメントだけで生じる。", ""]
    D = {}
    for act, seed in RUNS:
        f = OUT / f"bodystep_{act}_s{seed}_t{T}.npz"
        D[act] = np.load(f)
    L += ["## 完全一致の検査", "", "| 活性化 | 課題 24 末の θ と保存値の最大差 | H0 の更新 1 の体の勾配 |", "|---|---|---|"]
    for act, _ in RUNS:
        d = D[act]; L.append(f"| {NAME[act]} | {fmt(float(d['check_end_maxabs']))} | {fmt(float(d['H0_body_grad'][0]))} |")
    L.append("")
    for key, title in [("body_step", "体の変位の平均"), ("head_step", "head の変位の平均"), ("body_grad", "体の勾配の平均")]:
        L += [f"## {title}（更新の区間ごと）", "", "| 活性化 | 腕 | " + " | ".join(w[0] for w in W) + " |", "|---|---|" + "---|" * len(W)]
        for act, _ in RUNS:
            for arm in ARMS:
                a = D[act][f"{arm}_{key}"]
                L.append(f"| {NAME[act]} | {arm} | " + " | ".join(fmt(float(a[s:e].mean())) for _, s, e in W) + " |")
        L.append("")
    L += ["## 体の変位の比 H0 / H0A", "", "| 活性化 | 1–100 | 101–1,000 |", "|---|---|---|"]
    for act, _ in RUNS:
        d = D[act]
        r = [float(d["H0_body_step"][s:e].mean() / d["H0A_body_step"][s:e].mean()) for _, s, e in W[:2]]
        L.append(f"| {NAME[act]} | {fmt(r[0])} | {fmt(r[1])} |")
    L.append("")
    U = [1, 10, 100, 1000, 4000]
    L += ["## v の尺度（更新後）", "", "| 活性化 | 腕 | " + " | ".join(f"更新 {u:,}" for u in U) + " |", "|---|---|" + "---|" * len(U)]
    for act, _ in RUNS:
        for arm in ARMS:
            a = D[act][f"{arm}_v_body_rms"]
            L.append(f"| {NAME[act]} | {arm} | " + " | ".join(fmt(float(a[u - 1])) for u in U) + " |")
    L.append("")
    UA = [100, 1000, 4000]
    L += ["## 訓練精度", "", "| 活性化 | 腕 | 更新 0 | " + " | ".join(f"更新 {u:,}" for u in UA) + " |", "|---|---|---|" + "---|" * len(UA)]
    for act, _ in RUNS:
        for arm in ARMS:
            a = D[act][f"{arm}_acc"]
            L.append(f"| {NAME[act]} | {arm} | {a[0]:.3f} | " + " | ".join(f"{a[u // EVAL_EVERY]:.3f}" for u in UA) + " |")
    L += ["", f"生の配列は `results/head0_mlp_1108/posthoc/bodystep_<活性化>_s0_t{T}.npz`、図は `bodystep.png` にある。",
          "スクリプトは `analysis/head0_mlp_1108/posthoc_bodystep.py` である。", ""]
    (OUT / "bodystep.md").write_text("\n".join(L))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    col = {"S": "black", "H0": "#D55E00", "H0A": "#009E73"}
    fig, ax = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for j, (act, _) in enumerate(RUNS):
        d = D[act]
        for arm in ARMS:
            bs = d[f"{arm}_body_step"]
            rm = np.convolve(bs, np.ones(10) / 10, mode="valid")
            ax[0, j].plot(np.arange(10, len(bs) + 1), rm, color=col[arm], lw=0.8, label=arm)
            acc = d[f"{arm}_acc"]
            ax[1, j].plot(np.arange(len(acc)) * EVAL_EVERY, acc, color=col[arm], lw=0.8, label=arm)
        ax[0, j].set_yscale("log"); ax[0, j].set_title(NAME[act])
        ax[1, j].set_xlabel("更新"); ax[1, j].set_ylim(0, 1.02)
    ax[0, 0].set_ylabel("体の 1 更新の変位 ‖Δθ_body‖\n（10 更新の移動平均）"); ax[1, 0].set_ylabel("訓練精度")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"課題 {T} の切替後（seed 0）")
    fig.tight_layout()
    fig.savefig(OUT / "bodystep.png", dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[a for a, _ in RUNS])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.only:
        compute(a.only, dict(RUNS)[a.only]); return
    if not a.report_only:
        procs = [subprocess.Popen([sys.executable, "-m", "analysis.head0_mlp_1108.posthoc_bodystep", "--only", act],
                                  cwd=WORKTREE) for act, _ in RUNS]
        codes = [p.wait() for p in procs]
        if any(codes): sys.exit(f"子プロセスが失敗: {codes}")
    report()


if __name__ == "__main__":
    main()

"""箱 M の連鎖の性質を α ごとに見積もる（spec §4.3。学習なし）。

学習コード src/chargpt_cl_1019.py（ブランチ claude/chargpt_cl_1019 033679e0、blob cc524f85。
main の analysis/chargpt_latepush_1023/source/src/ に同じ blob の写しがある）の ChainM・task_data_M を、
学習と同じ乱数の stream（遷移表 np_gen(seed, "rule", task)、系列 cpu_gen(seed, "data", task) の最初の 2 塊 = 500 更新）で呼び、次を出す。
  - 連鎖: H（π で重みをつけた条件付き entropy、nats）、文脈ごとの最頻遷移の確率 max p の分布、
    E_π[1 − max p]（ラベルが最頻の記号でない割合）、E_π[1 − Σp²]（完全に当てはめた網での ‖p − y‖² の期待値）
  - π: 有効な状態数 exp(H(π))、質量の 50/90/99% を持つ状態の数、収束（pi_iters、最後の 1 歩の L1 の変化）
  - 最頻の写像 f(a,b) = (b, argmax_c P(c|a,b)) の閉路（一様な出発点からの閉路長と ρ 長、閉路上の状態の数）
  - 系列（最初の 500 更新 = 16,000 本）: 最頻でないラベルの割合、異なる文脈の数、同じ系列で同じ文脈が前に出ていた目標の割合と、
    その前回の続きを写すと当たる割合（文脈の中の写し取りで当たる割合）
使い方: python -I chain_stats.py <out_json> <alpha> <seed> <task> [--no-pi]
  学習コードの場所は環境変数 DETLABEL_SRC（既定は <repo>/analysis/chargpt_latepush_1023/source/src）。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("DETLABEL_SRC", str(REPO / "analysis/chargpt_latepush_1023/source/src")))
import torch  # noqa: E402

torch.set_num_threads(1)
import chargpt_cl_1019 as C  # noqa: E402

V = C.V


def argmax_cycles(P2):
    """P2: (V*V, V)。最頻の写像の閉路の構造。返り値は一様な出発点ごとの閉路長・ρ 長と、閉路の一覧。"""
    N = V * V
    s = np.arange(N)
    f = (s % V) * V + P2.argmax(1)
    state = np.zeros(N, np.int8)         # 0 未訪問、1 訪問中、2 済み
    cyc_len = np.zeros(N, np.int64)      # 行き着く閉路の長さ
    dist = np.zeros(N, np.int64)         # 閉路までの距離（閉路上なら 0）
    on_cycle = np.zeros(N, bool)
    cyc_id = np.full(N, -1, np.int64)
    cycles = []
    for s0 in range(N):
        if state[s0]:
            continue
        path = []
        x = s0
        while state[x] == 0:
            state[x] = 1
            path.append(x)
            x = f[x]
        if state[x] == 1:                # 新しい閉路
            k = path.index(x)
            cyc = path[k:]
            L = len(cyc)
            cid = len(cycles)
            cycles.append(L)
            for y in cyc:
                cyc_id[y] = cid
                on_cycle[y] = True
                cyc_len[y] = L
                dist[y] = 0
                state[y] = 2
            tail = path[:k]
        else:
            tail = path
        # tail を後ろから埋める
        for y in reversed(tail):
            nx = f[y]
            cyc_len[y] = cyc_len[nx]
            cyc_id[y] = cyc_id[nx]
            dist[y] = dist[nx] + 1
            state[y] = 2
    rho = dist + cyc_len                 # 出発点から最初の繰り返しまでの異なる状態の数
    basin = np.bincount(cyc_id, minlength=len(cycles)) / N      # 一様な出発点が流れ込む割合
    return dict(f=f, cyc_len=cyc_len, dist=dist, on_cycle=on_cycle, cycles_raw=np.array(cycles), basin=basin,
                cycles=sorted(cycles, reverse=True), rho=rho)


def q(v, ps=(0.1, 0.5, 0.9)):
    return [float(x) for x in np.quantile(v, ps)]


def main():
    out, alpha, seed, task = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    no_pi = "--no-pi" in sys.argv
    res = dict(alpha=alpha, seed=seed, task=task)
    if no_pi:
        rng = C.np_gen(seed, "rule", task)
        P2 = rng.dirichlet(alpha * np.ones(V), size=V * V)
        ac = argmax_cycles(P2)
        res.update(cycles=ac["cycles"], n_cycles=len(ac["cycles"]), n_cyclic=int(ac["on_cycle"].sum()),
                   cyc_len_uniform_mean=float(ac["cyc_len"].mean()), cyc_len_uniform_q=q(ac["cyc_len"]),
                   rho_uniform_mean=float(ac["rho"].mean()), rho_uniform_q=q(ac["rho"]),
                   dist_uniform_mean=float(ac["dist"].mean()))
        # 最頻の写像だけの連鎖を学習コードの規則（π = 一様からの遅延連鎖の極限）で回すと、
        # 閉路 k に流域の割合 w_k の質量が乗り、系列は周期 L_k で繰り返す（出発点は閉路の上）
        w, Lk = ac["basin"], ac["cycles_raw"]
        res.update(basin_weighted_period=float((w * Lk).sum()), basin_weighted_period_q=[int(np.sort(Lk)[np.searchsorted(np.cumsum(w[np.argsort(Lk, kind='stable')]), p_)]) for p_ in (0.1, 0.5, 0.9)],
                   copyable_frac_basin=float((w * np.maximum(0, 127 - Lk) / 127).sum()),
                   frac_seq_period_le_64=float(w[Lk <= 64].sum()))
        json.dump(res, open(out, "w"))
        return
    t0 = time.time()
    ch = C.ChainM(seed, task, alpha)
    res["chain_sec"] = time.time() - t0
    res["pi_iters"] = int(ch.pi_iters)
    P = ch.P
    P2 = P.reshape(V * V, V)
    pi = ch.pi.reshape(-1)
    new = 0.5 * ch.pi + 0.5 * np.einsum("ab,abc->bc", ch.pi, P)
    res["pi_last_step_L1"] = float(np.abs(new - ch.pi).sum())
    res["H"] = ch.H
    hrow = ch.hrow.reshape(-1)
    mx = P2.max(1)
    srt = np.sort(P2, 1)
    second = srt[:, -2]
    coll = (P2 ** 2).sum(1)
    res.update(H_uniform_ctx=float(hrow.mean()),
               maxp_uniform_q=q(mx, (0.1, 0.25, 0.5, 0.75, 0.9)), maxp_uniform_mean=float(mx.mean()),
               maxp_pi_mean=float((pi * mx).sum()),
               nonmode_pi=float((pi * (1 - mx)).sum()), nonmode_uniform=float((1 - mx).mean()),
               resid_pi=float((pi * (1 - coll)).sum()), resid_uniform=float((1 - coll).mean()),
               second_uniform_q=q(second, (0.5, 0.9, 0.99)),
               frac_rows_exact_onehot=float((mx == 1.0).mean()), frac_zero_entries=float((P2 == 0).mean()),
               rowsum_err=ch.rowsum_err)
    # π の広がり
    ps = np.sort(pi)[::-1]
    cs = np.cumsum(ps)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hpi = -np.sum(np.where(pi > 0, pi * np.log(np.where(pi > 0, pi, 1)), 0))
    res.update(pi_eff_states=float(np.exp(Hpi)), pi_n50=int(np.searchsorted(cs, 0.5) + 1),
               pi_n90=int(np.searchsorted(cs, 0.9) + 1), pi_n99=int(np.searchsorted(cs, 0.99) + 1),
               pi_support=int((pi > 0).sum()))
    # 最頻の写像
    ac = argmax_cycles(P2)
    res.update(cycles=ac["cycles"][:20], n_cycles=len(ac["cycles"]), n_cyclic=int(ac["on_cycle"].sum()),
               cyc_len_uniform_mean=float(ac["cyc_len"].mean()), rho_uniform_mean=float(ac["rho"].mean()),
               pi_mass_on_argmax_cycles=float(pi[ac["on_cycle"]].sum()))
    # 系列（task_data_M と同じ stream の最初の 2 塊 = 500 更新 = 16,000 本）
    gen = C.cpu_gen(seed, "data", task)
    seqs = [ch.sample(C.M_CHUNK * C.M_BATCH, gen) for _ in range(2)]
    x = torch.cat(seqs).numpy().astype(np.int64)            # (16000, 129)
    ctx = x[:, :-2] * V + x[:, 1:-1]                         # 目標 s_2..s_128 の文脈
    tg = x[:, 2:]
    mode = P2.argmax(1)
    res["label_nonmode_frac_data"] = float((tg != mode[ctx]).mean())
    res["true_ce_data"] = float(-np.log(np.maximum(P2[ctx, tg], 1e-300)).mean())
    res["distinct_ctx_500upd"] = int(np.unique(ctx).size)
    res["distinct_ctx_100upd"] = int(np.unique(ctx[:3200]).size)
    res["distinct_ctx_per_seq_mean"] = float(np.mean([np.unique(r).size for r in ctx[:2000]]))
    n = ctx.shape[0]
    last = np.full((n, V * V), -1, np.int8)
    seen = np.zeros_like(tg, bool)
    corr = np.zeros_like(tg, bool)
    ar = np.arange(n)
    for j in range(ctx.shape[1]):
        prev = last[ar, ctx[:, j]]
        seen[:, j] = prev >= 0
        corr[:, j] = prev == tg[:, j]
        last[ar, ctx[:, j]] = tg[:, j]
    res["ctx_seen_before_frac"] = float(seen.mean())
    res["copy_correct_frac"] = float(corr.mean())               # 目標のうち、前回の続きを写して当たる割合
    res["copy_correct_given_seen"] = float(corr.sum() / max(seen.sum(), 1))
    res["copy_correct_frac_second_half"] = float(corr[:, 63:].mean())
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()

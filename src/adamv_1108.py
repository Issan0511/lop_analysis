#!/usr/bin/env python3
"""adamv_1108: Adam の v の記憶の向き（spec specs/spec_head0_mlp_1108.md 追補 C）の MLP 再生と記録。

src/head0_mlp_1108.py の engine 再生（ビット一致を検査済み）をそのまま使い、本線 S の全更新と、
課題 t ≥ 2 の頭の snap から回す枝 H0・H0A（各 branch_steps 更新、engine のミニバッチ順・同じラベル）に
体（W1・b1、78,500 座標）の Adam の記録を付ける。記録の読みは no_grad で乱数に触れず、
forward・loss・backward・step の演算は engine と同じ。本線の課題末 θ は保存値 end*[t-1] と完全一致を検査する。

系列の列（更新 n = 1..、課題の中で 1 始まり）:
  0 rms_g   : 更新 n のミニバッチ勾配（step の前）の体の RMS
  1 rms_v   : step の前の体の v̂ = exp_avg_sq/(1−β2^step) の √mean（状態が無ければ 0）
  2 R       : rms_g / rms_v（rms_v = 0 なら inf）
  3 med     : 座標ごとの |g_i| / (√v̂_i + 1e-16) の中央値（状態が無ければ NaN）
  4 D       : 体の変位の RMS ÷ lr
  5 rms_g_head : head（W2・b2）の勾配の RMS
  6 loss    : ミニバッチの CE（更新の前）
head の変位の RMS ÷ lr は別の配列 *_Dhead、体の勾配が 0 でない座標の割合は *_nz（Adam の状態が無い一歩の D は √nz で、1 ではない）。
"""
import argparse, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head0_mlp_1108 import (load_X, stream, initial, order_for, new_params, new_adam, snap, restore, activ,  # noqa: E402
                            evaluate, unit_stats, Ctx, maxabs, ENGINE_STEPS, K, N, H, SAVED, LR, BETAS)

torch.set_num_threads(1)
WORKTREE = Path(__file__).resolve().parent.parent
NCOL = 7
COLS = ["rms_g", "rms_v", "R", "med", "D", "rms_g_head", "loss"]
DEAD = 0.005


def make_h0(ck, fresh_adam=False):
    """head0_mlp_1108.main() の make_h0(keep_head_adam=False, fresh_adam) と同じ。"""
    if fresh_adam:
        ck2 = dict(ck); ck2.update({f"step{q}": np.array(0.0) for q in range(4)})
        Pb, ob = restore(ck2)
    else:
        Pb, ob = restore(ck)
    with torch.no_grad():
        Pb[2].data.zero_(); Pb[3].data.zero_()
    if not fresh_adam:
        ob.state.pop(Pb[2], None); ob.state.pop(Pb[3], None)
    return Pb, ob


def _body_vhat(optim, P):
    s0, s1 = optim.state.get(P[0]), optim.state.get(P[1])
    if not s0 or not s1 or "exp_avg_sq" not in s0 or "exp_avg_sq" not in s1:
        return None, 0.0
    st = float(s0["step"])
    assert st == float(s1["step"]) and st > 0
    bc = 1.0 - BETAS[1] ** st                 # torch.optim.Adam と同じ bias_correction2（step の前なので更新 n−1 まで）
    v = torch.cat([s0["exp_avg_sq"].reshape(-1), s1["exp_avg_sq"].reshape(-1)]).double() / bc
    return v, st


def run_inst(P, optim, ctx, y, Y, order, steps, eval_at=(100, 1000), first_info=None):
    """engine の step() と同じ演算で steps 更新回し、更新ごとの記録と、更新 0・100・1,000（・末）の p⁺ と精度を返す。"""
    act, X = ctx.act, ctx.X
    ser = np.full((steps, NCOL), np.nan, dtype=np.float64)
    dhead = np.full(steps, np.nan, dtype=np.float64)
    nz = np.full(steps, np.nan, dtype=np.float64)
    pp = {0: unit_stats(P, ctx)[0][1].copy()}
    acc = {}
    for it in range(steps):
        idx = order[it]
        xb, yb = X[idx], Y[idx]
        z = xb @ P[0].T + P[1]; a = activ(z, act); logits = a @ P[2].T + P[3]
        loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
        optim.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            g = torch.cat([P[0].grad.reshape(-1), P[1].grad.reshape(-1)]).double()
            rms_g = math.sqrt(float(g.pow(2).mean()))
            nz[it] = float((g != 0).double().mean())
            vhat, st = _body_vhat(optim, P)
            if vhat is None:
                rms_v, med = 0.0, float("nan")
            else:
                rms_v = math.sqrt(float(vhat.mean()))
                # v̂ が全座標で 0（H0A の更新 2: 更新 1 の勾配が 0）のときは状態が無いのと同じ扱いで NaN
                med = float(torch.median(g.abs() / (vhat.sqrt() + 1e-16))) if rms_v > 0 else float("nan")
            R = rms_g / rms_v if rms_v > 0 else float("inf")
            gh = torch.cat([P[2].grad.reshape(-1), P[3].grad.reshape(-1)]).double()
            rms_gh = math.sqrt(float(gh.pow(2).mean()))
            before = [q.detach().clone() for q in P]
            if first_info is not None and it == 0:
                gf = g.float()
                first_info["frac_coords_g_nonzero"] = float((gf != 0).double().mean())
                # Adam の状態が無い一歩の予想 RMS(Δθ)/lr = √mean((g/(|g|+eps))²)（float32 の演算で）
                first_info["D1_expected_if_no_state"] = math.sqrt(float((gf / (gf.abs() + 1e-8)).double().pow(2).mean()))
                first_info["state_step_before"] = st
        optim.step()
        with torch.no_grad():
            db = torch.cat([(P[0] - before[0]).reshape(-1), (P[1] - before[1]).reshape(-1)]).double()
            dh = torch.cat([(P[2] - before[2]).reshape(-1), (P[3] - before[3]).reshape(-1)]).double()
            D = math.sqrt(float(db.pow(2).mean())) / LR
            dhead[it] = math.sqrt(float(dh.pow(2).mean())) / LR
        ser[it] = (rms_g, rms_v, R, med, D, rms_gh, float(loss.detach()))
        n = it + 1
        if n in eval_at:
            pp[n] = unit_stats(P, ctx)[0][1].copy()
            acc[n] = float(evaluate(P, ctx, y)[0])
    acc["end"] = float(evaluate(P, ctx, y)[0])
    pp["end"] = unit_stats(P, ctx)[0][1].copy()
    return ser, dhead, nz, pp, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True, choices=["LR", "ELU", "R", "GELU", "SILU"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results/head0_mlp_1108/adamv/raw")
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--branch-steps", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.tasks, a.branch_steps, a.out = 3, 300, "results/_smoke/adamv_1108/raw"
    out = Path(a.out)
    if not out.is_absolute(): out = WORKTREE / out
    act, seed, tasks, bsteps = a.act, a.seed, a.tasks, a.branch_steps
    steps = ENGINE_STEPS
    assert 1 <= bsteps <= steps
    t_start = time.monotonic()

    X, subset = load_X(seed)
    saved = np.load(SAVED / f"{act}_K10_s{seed}.npz")
    assert np.array_equal(saved["subset"], subset), "subset が保存済みと違う"
    n_saved = saved["y"].shape[0]
    assert tasks <= n_saved
    ctx = Ctx(X, act)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    P = new_params(initial(seed, (784, H, K)))
    optim = new_adam(P)
    init_ref = [saved[f"init{q}"] for q in range(4)]
    check = dict(init_equal=maxabs(P, init_ref) == 0.0,
                 frac_pixels_zero_in_all_1200=float((X.abs().sum(0) == 0).double().mean()))

    S_series = np.full((tasks, steps, NCOL), np.nan, np.float32); S_Dhead = np.full((tasks, steps), np.nan, np.float32)
    H0_series = np.full((tasks - 1, bsteps, NCOL), np.nan, np.float32); H0_Dhead = np.full((tasks - 1, bsteps), np.nan, np.float32)
    H0A_series = np.full_like(H0_series, np.nan); H0A_Dhead = np.full_like(H0_Dhead, np.nan)
    S_nz = np.full_like(S_Dhead, np.nan); H0_nz = np.full_like(H0_Dhead, np.nan); H0A_nz = np.full_like(H0_Dhead, np.nan)
    pplus_S = np.full((tasks, 4, H), np.nan, np.float32)
    pplus_H0 = np.full((tasks - 1, 3, H), np.nan, np.float32); pplus_H0A = np.full_like(pplus_H0, np.nan)
    acc_S = np.full((tasks, 3), np.nan, np.float32)
    acc_H0 = np.full((tasks - 1, 3), np.nan, np.float32); acc_H0A = np.full_like(acc_H0, np.nan)
    check_end = np.full(tasks, np.nan)
    per_task = []

    for t in range(1, tasks + 1):
        t_task = time.monotonic()
        y = torch.randint(K, (N,), generator=glabel)
        order = order_for(gbatch, steps)
        assert np.array_equal(y.numpy(), saved["y"][t - 1]), f"課題 {t} のラベルが保存済みと違う"
        Y = torch.nn.functional.one_hot(y, K).to(torch.float32)

        if t >= 2:
            ck = snap(P, optim, (t - 1) * steps, float("nan"))
            for name, fresh, ser_a, dh_a, nz_a, pp_a, acc_a in (("H0", False, H0_series, H0_Dhead, H0_nz, pplus_H0, acc_H0),
                                                                ("H0A", True, H0A_series, H0A_Dhead, H0A_nz, pplus_H0A, acc_H0A)):
                Pb, ob = make_h0(ck, fresh_adam=fresh)
                fi = {} if t == 2 else None
                ser, dh, nzv, pp, acc = run_inst(Pb, ob, ctx, y, Y, order, bsteps, first_info=fi)
                if fi is not None: check[f"{name}_t02_first"] = fi
                ser_a[t - 2] = ser; dh_a[t - 2] = dh; nz_a[t - 2] = nzv
                for j, k in enumerate((0, 100, 1000)):
                    if k in pp: pp_a[t - 2, j] = pp[k]
                for j, k in enumerate((100, 1000)):
                    if k in acc: acc_a[t - 2, j] = acc[k]
                acc_a[t - 2, 2] = acc["end"]          # 枝の末 = 更新 branch_steps
            del ck

        fi = {} if t == 1 else None
        ser, dh, nzv, pp, acc = run_inst(P, optim, ctx, y, Y, order, steps, first_info=fi)
        if fi is not None: check["S_t01_first"] = fi
        S_series[t - 1] = ser; S_Dhead[t - 1] = dh; S_nz[t - 1] = nzv
        for j, k in enumerate((0, 100, 1000, "end")): pplus_S[t - 1, j] = pp[k]
        acc_S[t - 1] = (acc[100], acc[1000], acc["end"])
        check_end[t - 1] = maxabs(P, [saved[f"end{q}"][t - 1] for q in range(4)])
        el = time.monotonic() - t_task
        r15 = float(np.mean(ser[:5, 2]))
        msg = f"{act} s{seed} task {t:2d} S acc_end {acc['end']:.3f} (saved {float(saved['acc'][t-1]):.3f}) end_maxabs {check_end[t-1]} " \
              f"S R(1-5) {r15:.3g} D(1-5) {float(np.mean(ser[:5, 4])):.3g}"
        if t >= 2:
            msg += f" | H0 R(2-5) {float(np.mean(H0_series[t-2, 1:5, 2])):.3g} D(1-5) {float(np.mean(H0_series[t-2, :5, 4])):.3g}" \
                   f" | H0A D(1) {float(H0A_series[t-2, 0, 4]):.6f}"
        print(msg + f" ({el:.1f}s, total {time.monotonic()-t_start:.0f}s)", flush=True)
        per_task.append(dict(task=t, elapsed_s=round(el, 2), S_acc_end=acc["end"], saved_acc=float(saved["acc"][t - 1])))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{act}_s{seed}.npz", S_series=S_series, H0_series=H0_series, H0A_series=H0A_series,
                        S_Dhead=S_Dhead, H0_Dhead=H0_Dhead, H0A_Dhead=H0A_Dhead, S_nz=S_nz, H0_nz=H0_nz, H0A_nz=H0A_nz,
                        pplus_S=pplus_S, pplus_H0=pplus_H0, pplus_H0A=pplus_H0A,
                        acc_S=acc_S, acc_H0=acc_H0, acc_H0A=acc_H0A, check_end_maxabs=check_end,
                        y=saved["y"][:tasks].copy())
    try:
        head = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        head = f"error: {e}"
    # 枝の検査値（smoke の確認項目）
    if tasks >= 2:
        check["H0_t02_R1"] = float(H0_series[0, 0, 2]); check["H0_t02_D1"] = float(H0_series[0, 0, 4])
        check["H0_t02_rms_g1"] = float(H0_series[0, 0, 0])
        check["H0A_t02_rms_v1"] = float(H0A_series[0, 0, 1]); check["H0A_t02_D1"] = float(H0A_series[0, 0, 4])
    check["S_t01_D1"] = float(S_series[0, 0, 4]); check["S_t01_rms_v1"] = float(S_series[0, 0, 1])
    check["S_t01_D1_minus_expected"] = check["S_t01_D1"] - check["S_t01_first"]["D1_expected_if_no_state"]
    check["note_D1"] = ("Adam の状態が無い一歩の D は √mean((g/(|g|+eps))²) ≈ √(勾配が 0 でない座標の割合) で、1 にはならない"
                        "（1,200 枚すべてで 0 の画素の列とミニバッチ 16 枚で 0 の画素の列は勾配が 0）。"
                        "H0A は head が 0 なので更新 1 の体の勾配が 0 で D(1) = 0、R(1)・R(2) は v̂ = 0 で inf。")
    check["check_end_maxabs"] = check_end.tolist()
    meta = dict(act=act, seed=seed, tasks=tasks, steps=steps, branch_steps=bsteps, smoke=a.smoke, columns=COLS,
                pplus_points=dict(S=[0, 100, 1000, "end"], branches=[0, 100, 1000]),
                acc_points=dict(S=[100, 1000, "end"], branches=[100, 1000, "end(=branch_steps)"]),
                dead_threshold=DEAD, lr=LR, betas=list(BETAS), n_body=784 * H + H,
                torch=torch.__version__, git_head=head,
                engine_sha256="4571091d4c02ef6a3d74457a62ad128f2ae177525d8a48d47f2ed70b0fec3f30",
                elapsed_s=round(time.monotonic() - t_start, 1), per_task=per_task, check=check)
    (out / f"{act}_s{seed}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(dict(act=act, seed=seed, check=check), ensure_ascii=False), flush=True)

    assert check["init_equal"], "engine の初期値と一致しない"
    assert np.all(check_end == 0.0), f"C1 end 不一致 {check_end}"
    if tasks >= 2:
        assert check["H0_t02_rms_g1"] == 0.0, "H0 の更新 1 の体の勾配が 0 でない"
        assert check["H0A_t02_rms_v1"] == 0.0
        assert check["H0A_t02_D1"] == 0.0, "H0A の更新 1 は体の勾配が 0 なので変位も 0 のはず"
    assert abs(check["S_t01_D1_minus_expected"]) < 1e-6, "S 課題 1 の更新 1 の D が sign の一歩の予想と違う"
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()

"""rank_int_0814 Phase 1: ランク介入実験のランナー (spec_rank_int_0814 §3–5)。

  python -m src.rank_int --config configs/rank_int_0814.yaml [--widths 10 20]
                         [--smoke] [--device cpu]

幅ごとの流れ:
  1. 連続 run 0→350k (= none アーム、pre 区間共有)。t_int=150k で完全スナップショット
  2. S1: スナップショットから介入なし resume (150k→350k)。lop_metrics / online_loss の
     step ≥ t_int 行が連続 run と**文字列レベルで全行一致**することを要求。
     不一致なら即 abort (S1 が通るまで本体に進まない)
  3. svdrec / shuffle: スナップショットの W を介入で差し替えて resume (+200k)。
     処理順は「介入 → タスク切替 → 学習継続」(t_int はタスク境界、maybe_flip は
     resume 後最初の step の冒頭で発火するため、この順序が自動的に成立する)
  4. intervention_log.csv と S2 サニティを出力

介入の数値計算は float64 で行い、学習再開時に float32 へ戻す (S2 の相対 1e-6 を
float32 の SVD 再構成誤差で破らないため)。
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .common import (ROOT, load_config, pick_device, build_runs, group_runs,
                     group_name, resolve_outdir)
from .train import (setup_group, train_group, load_resume, eval_batch)
from .lop_metrics import compute_lop_metrics


# ---------------------------------------------------------------- 介入 (float64)

def dead_mask_and_metrics(st, cfg):
    """介入前後の計測: eval バッチでの dead マスク [R,h] と主要メトリクス。"""
    x_ev, y_ev = eval_batch(st)
    cmask = st["centered"][:, None].float()
    x_in = x_ev - cmask[None] * st["running_mean"][None]
    m = compute_lop_metrics(st["net"], x_in, y_ev, cfg)
    C = cfg["common"]
    with torch.no_grad():
        pre, a, _ = st["net"].forward_batch(x_in)
        dead_i = ((a.abs() < C["dead_tol"]).float().mean(dim=0) > C["dead_tau"])  # [R,h]
        n_gate_open = (pre > 0).any(dim=0).sum(dim=1)                              # [R]
    return dead_i, dict(stable_rank_W_alive=m["stable_rank_W_alive"].cpu().numpy(),
                        dead_frac=m["dead_frac"].cpu().numpy(),
                        n_gate_open=n_gate_open.cpu().numpy())


def srank_alive64(W, dead_row):
    """float64 W [h,d] の dead 行ゼロ化 stable rank。"""
    Wa = W.clone()
    Wa[dead_row] = 0.0
    s2 = torch.linalg.svdvals(Wa) ** 2
    return float(s2.sum() / s2.max().clamp_min(1e-300))


def intervene_svdrec(W, dead_row, target, eps_lo, srank_tol):
    """条件 B: s'_i = max(s_i, ε·s_1) → Frobenius 一様再スケール → W' = U S'' Vᵀ。
    ε は介入直後の stable_rank_W_alive ≈ target となるよう bisect (ε に単調増加)。
    target 到達不能 (ε=1 でも下回る) なら ε=1 で打ち切り clipped=True。"""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    normF = S.norm()

    def apply_eps(eps):
        Sp = torch.clamp(S, min=eps * S[0])
        Sp = Sp * (normF / Sp.norm())
        return U @ torch.diag(Sp) @ Vh

    clipped = False
    if srank_alive64(apply_eps(1.0), dead_row) < target:
        eps, clipped = 1.0, True
    else:
        lo, hi = eps_lo, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if srank_alive64(apply_eps(mid), dead_row) < target:
                lo = mid
            else:
                hi = mid
            if (hi - lo) / hi < srank_tol * 1e-2:
                break
        eps = hi                      # target 以上側 (bisect の上端) を採用
    Wp = apply_eps(eps)
    return Wp, dict(eps=eps, eps_clipped=clipped, dF=float((Wp - W).norm()))


def intervene_shuffle(W, dF_target, energy_frac, gen, match_tol):
    """条件 C: top-k (エネルギー energy_frac) 部分空間内のランダム回転
    W'(θ) = U_k Q(θ) S_k V_kᵀ + 残差。特異値・両基底 span・‖W‖_F・ランク不変。
    θ を bisect して ‖W'−W‖_F = dF_target (相対 match_tol)。

    ‖W'−W‖_F = ‖(Q(θ)−I) S_k‖_F は G の回転角 θλ_j (λ_max ≤ ‖G‖_F/√2 < 1) が
    θ ≤ π で全て π 未満のため θ に単調増加 → bisect 可。θ=π で届かなければ abort。"""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = int(torch.searchsorted(energy, energy_frac).item()) + 1
    k = max(k, 2)                     # 回転には最低 2 次元必要
    Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k]
    resid = W - Uk @ torch.diag(Sk) @ Vhk

    # G の抽選: dF(π) < dF_target なら同一 generator 列から再抽選 (決定論的な棄却
    # サンプリング、最大 50 回)。仕様の「seed 固定」の最小拡張 (summary に逸脱として
    # 明記)。G の回転角スペクトルによっては単発抽選で僅かに届かないことがある
    # (実例: w20/s4 で 9.08 < 9.24)。
    n_draw = 0
    while True:
        A = torch.randn(k, k, generator=gen, device=W.device).to(W.dtype)
        G = A - A.T
        G = G / G.norm()
        n_draw += 1

        def dF(theta):
            Q = torch.matrix_exp(theta * G)
            return float(((Q - torch.eye(k, dtype=W.dtype, device=W.device))
                          @ torch.diag(Sk)).norm())

        if dF(torch.pi) >= dF_target:
            break
        if n_draw >= 50:
            return None, dict(k=k, theta=np.nan, dF=dF(torch.pi), n_draw=n_draw,
                              abort=f"unreachable after {n_draw} draws: "
                                    f"dF(pi)={dF(torch.pi):.4g} < {dF_target:.4g}")
    lo, hi = 0.0, float(torch.pi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dF(mid) < dF_target:
            lo = mid
        else:
            hi = mid
        if dF_target > 0 and (dF(hi) - dF_target) / dF_target < match_tol and \
           (dF_target - dF(lo)) / dF_target < match_tol:
            break
    theta = 0.5 * (lo + hi)
    Q = torch.matrix_exp(theta * G)
    Wp = Uk @ Q @ torch.diag(Sk) @ Vhk + resid
    return Wp, dict(k=k, theta=theta, dF=float((Wp - W).norm()), n_draw=n_draw, abort="")


def s2_checks(W, Wp_svd, Wp_shf, tol):
    """S2: (i) shuffle の特異値保存 (ii) ΔF 一致 (iii) 両アームの ‖W'‖_F 保存。"""
    s0 = torch.linalg.svdvals(W)
    checks = {}
    if Wp_shf is not None:
        s1 = torch.linalg.svdvals(Wp_shf)
        checks["sv_preserved"] = float(((s1 - s0).abs().max() / s0[0]))
        dfs = float((Wp_svd - W).norm())
        dfh = float((Wp_shf - W).norm())
        checks["dF_match"] = abs(dfh - dfs) / max(dfs, 1e-300)
        checks["normF_shuffle"] = float(abs(Wp_shf.norm() - W.norm()) / W.norm())
    checks["normF_svdrec"] = float(abs(Wp_svd.norm() - W.norm()) / W.norm())
    checks["pass"] = all(v < tol for kk, v in checks.items() if kk != "pass")
    return checks


def intervene_svdrec_alive(W, dead_row, target, eps_lo, srank_tol):
    """G1 破れ時の感度分析 (spec §6): SVD 介入を alive 行の部分行列のみに適用し、
    dead 行は不変のまま残す (dead 行のゲート状態を介入が触らない)。
    alive 部分行列の stable rank = dead 行ゼロ化 srank と等価なので target 規則は同一。"""
    alive = ~dead_row
    if int(alive.sum()) < 2:
        return None, dict(eps=np.nan, eps_clipped=True, dF=np.nan,
                          abort="fewer than 2 alive rows")
    Ws = W[alive]
    no_dead = torch.zeros(Ws.shape[0], dtype=torch.bool, device=W.device)
    Wps, info = intervene_svdrec(Ws, no_dead, target, eps_lo, srank_tol)
    Wp = W.clone()
    Wp[alive] = Wps
    info["dF"] = float((Wp - W).norm())
    info["abort"] = ""
    return Wp, info


# ---------------------------------------------------------------- S1 比較

def compare_logs(outdir, gname_a, gname_b, t_min):
    """lop_metrics / online_loss / postswitch_err の step ≥ t_min 行を、run_id の
    アーム接尾辞を除去した上で文字列比較。差異のあるファイル・行数を返す。"""
    diffs = {}
    # online_loss は step > t_min (連続 run の step=t_min ビンは resume 側に存在しない)。
    # lop_metrics は resume 側が step=t_min の初期行を持つため >= で比較できる。
    for prefix, stepcol, strict in [("lop_metrics", "step", False),
                                    ("online_loss", "step", True),
                                    ("postswitch_err", "switch_step", False)]:
        fa = os.path.join(outdir, f"{prefix}_{gname_a}.csv")
        fb = os.path.join(outdir, f"{prefix}_{gname_b}.csv")
        da, db = pd.read_csv(fa, dtype=str), pd.read_csv(fb, dtype=str)
        for d, g in [(da, gname_a), (db, gname_b)]:
            arm = g.rsplit("_", 1)[1]
            d["run_id"] = d.run_id.str.replace(f"_{arm}$", "", regex=True)
        da = da[da[stepcol].astype(int) > t_min if strict
                else da[stepcol].astype(int) >= t_min].reset_index(drop=True)
        db = db[db[stepcol].astype(int) > t_min if strict
                else db[stepcol].astype(int) >= t_min].reset_index(drop=True)
        if len(da) != len(db):
            diffs[prefix] = f"row count {len(da)} vs {len(db)}"
            continue
        neq = (da != db).any(axis=1).sum()
        if neq:
            diffs[prefix] = f"{neq}/{len(da)} rows differ"
    return diffs


# ---------------------------------------------------------------- ランナー

def arm_runs(base_runs, arm):
    out = []
    for r in base_runs:
        r2 = dict(r)
        r2["arm"] = arm
        r2["run_id"] = f"{r['run_id']}_{arm}"
        out.append(r2)
    return out


def run_width(width, cfg, device, outdir, targets, smoke=False):
    RI = cfg["rank_int"]
    t_int, post = RI["t_int"], RI["post_steps"]
    total = t_int + post
    base_runs = [r for r in build_runs(cfg) if r["width"] == width]
    gkey = ("A", width, "full", "none")
    assert group_runs(base_runs) and len(base_runs) == len(cfg["common"]["seeds"])
    gbase = group_name(gkey)          # A_w10_bfull

    log = {"width": width}
    # --- 1. 連続 run (= none アーム) + スナップショット
    gname_none = f"{gbase}_none"
    print(f"=== [{gbase}] continuous none 0->{total}", flush=True)
    st, el = train_group(gkey, arm_runs(base_runs, "none"), cfg, device, outdir,
                         total_steps=total, ckpts=[], gname=gname_none,
                         snapshot_steps=[t_int])
    print(f"    done {el:.1f}s ({total/el:.0f} steps/s)", flush=True)
    snap_path = os.path.join(outdir, "snapshots", f"{gname_none}_step{t_int}.pt")
    snap = torch.load(snap_path, weights_only=False)

    # --- 2. S1: 介入なし resume の bit 一致
    gname_s1 = f"{gbase}_s1resume"
    print(f"=== [{gbase}] S1 resume {t_int}->{total}", flush=True)
    train_group(gkey, arm_runs(base_runs, "s1resume"), cfg, device, outdir,
                total_steps=total, ckpts=[], gname=gname_s1,
                start_step=t_int, resume_state=snap)
    diffs = compare_logs(outdir, gname_none, gname_s1, t_int)
    log["S1"] = "PASS" if not diffs else f"FAIL: {diffs}"
    print(f"    S1: {log['S1']}", flush=True)
    if diffs:
        raise SystemExit(f"S1 FAILED ({gbase}): {diffs} — Phase 1 本体を中止")

    # --- 3. 介入計算 (float64)
    st_tmp = setup_group(gkey, base_runs, cfg, device)
    load_resume(st_tmp, snap)
    dead_i, pre_m = dead_mask_and_metrics(st_tmp, cfg)
    W32 = st_tmp["net"].W                              # [R,h,d]
    ilog_rows, s2_all = [], []
    W_svd = torch.empty_like(W32)
    W_shf = torch.empty_like(W32)
    for i, r in enumerate(base_runs):
        seed = r["seed"]
        W = W32[i].double()
        target = float(targets.loc[(width, seed), "srank_target"])
        Wp_svd, info_svd = intervene_svdrec(W, dead_i[i], target,
                                            RI["eps_lo"], RI["srank_tol"])
        gen = torch.Generator(device=W32.device)
        gen.manual_seed(RI["shuffle_seed_base"] + 1000 * width + seed)
        Wp_shf, info_shf = intervene_shuffle(W, info_svd["dF"], RI["energy_frac"],
                                             gen, RI["match_tol"])
        if Wp_shf is None:
            raise SystemExit(f"shuffle abort (w{width}/s{seed}): {info_shf['abort']}")
        s2 = s2_checks(W, Wp_svd, Wp_shf, RI["match_tol"])
        s2_all.append(s2)
        W_svd[i] = Wp_svd.float()
        W_shf[i] = Wp_shf.float()
        ilog_rows.append(dict(seed=seed, width=width, t_int=t_int,
                              srank_target=target, **{f"svd_{k}": v for k, v in info_svd.items()},
                              **{f"shf_{k}": v for k, v in info_shf.items()},
                              **{f"s2_{k}": v for k, v in s2.items()}))
    log["S2"] = "PASS" if all(s["pass"] for s in s2_all) else "FAIL"
    print(f"    S2: {log['S2']}", flush=True)

    # --- 4. 介入アームの resume (+介入前後メトリクス)
    for arm, Wnew in [("svdrec", W_svd), ("shuffle", W_shf)]:
        snap_arm = copy.deepcopy(snap)
        snap_arm["net"]["W"] = Wnew.clone()
        # 介入直後メトリクス (同じ eval バッチ、タスク切替前)
        st_tmp["net"].load_state(snap_arm["net"])
        _, post_m = dead_mask_and_metrics(st_tmp, cfg)
        for i, r in enumerate(base_runs):
            row = next(x for x in ilog_rows if x["seed"] == r["seed"])
            for k in ("stable_rank_W_alive", "dead_frac", "n_gate_open"):
                row[f"pre_{k}"] = float(pre_m[k][i])
                row[f"post_{arm}_{k}"] = float(post_m[k][i])
        gname = f"{gbase}_{arm}"
        print(f"=== [{gbase}] arm {arm} {t_int}->{total}", flush=True)
        _, el = train_group(gkey, arm_runs(base_runs, arm), cfg, device, outdir,
                            total_steps=total, ckpts=[], gname=gname,
                            start_step=t_int, resume_state=snap_arm)
        print(f"    done {el:.1f}s", flush=True)
    return ilog_rows, log


def run_extra_arm(width, cfg, device, outdir, targets, arm):
    """既存スナップショットから追加アームのみ実行 (感度分析用。none/S1 は再実行しない)。"""
    RI = cfg["rank_int"]
    t_int, post = RI["t_int"], RI["post_steps"]
    total = t_int + post
    base_runs = [r for r in build_runs(cfg) if r["width"] == width]
    gkey = ("A", width, "full", "none")
    gbase = group_name(gkey)
    snap = torch.load(os.path.join(outdir, "snapshots", f"{gbase}_none_step{t_int}.pt"),
                      weights_only=False)
    st_tmp = setup_group(gkey, base_runs, cfg, device)
    load_resume(st_tmp, snap)
    dead_i, pre_m = dead_mask_and_metrics(st_tmp, cfg)
    W32 = st_tmp["net"].W
    W_new = torch.empty_like(W32)
    rows = []
    for i, r in enumerate(base_runs):
        seed = r["seed"]
        target = float(targets.loc[(width, seed), "srank_target"])
        Wp, info = intervene_svdrec_alive(W32[i].double(), dead_i[i], target,
                                          RI["eps_lo"], RI["srank_tol"])
        if Wp is None:
            raise SystemExit(f"{arm} abort (w{width}/s{seed}): {info['abort']}")
        W_new[i] = Wp.float()
        rows.append(dict(seed=seed, width=width, t_int=t_int, arm=arm,
                         srank_target=target, **{f"svd_{k}": v for k, v in info.items()}))
    snap_arm = copy.deepcopy(snap)
    snap_arm["net"]["W"] = W_new.clone()
    st_tmp["net"].load_state(snap_arm["net"])
    _, post_m = dead_mask_and_metrics(st_tmp, cfg)
    for i, row in enumerate(rows):
        for k in ("stable_rank_W_alive", "dead_frac", "n_gate_open"):
            row[f"pre_{k}"] = float(pre_m[k][i])
            row[f"post_{arm}_{k}"] = float(post_m[k][i])
    gname = f"{gbase}_{arm}"
    print(f"=== [{gbase}] extra arm {arm} {t_int}->{total}", flush=True)
    _, el = train_group(gkey, arm_runs(base_runs, arm), cfg, device, outdir,
                        total_steps=total, ckpts=[], gname=gname,
                        start_step=t_int, resume_state=snap_arm)
    print(f"    done {el:.1f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rank_int_0814.yaml")
    ap.add_argument("--widths", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="t_int=20k, post=30k の短縮実行 (results/_smoke)")
    ap.add_argument("--extra-arm", default=None, choices=["svdrec_alive"],
                    help="既存スナップショットから感度分析アームのみ追加実行")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

    outdir = resolve_outdir(args.config, smoke=args.smoke)
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["rank_int"]["t_int"] = 20000
        cfg["rank_int"]["post_steps"] = 30000
    os.makedirs(outdir, exist_ok=True)
    print(f"outdir: {outdir}", flush=True)

    t0 = pd.read_csv(os.path.join(ROOT, "results", "rank_int_0814",
                                  "phase0_targets.csv"))
    targets = t0.set_index(["width", "seed"])

    cfg_total = cfg["rank_int"]["t_int"] + cfg["rank_int"]["post_steps"]
    cfg["common"]["total_steps"] = cfg_total

    widths = args.widths or cfg["condA"]["widths"]

    if args.extra_arm:
        rows = []
        for width in widths:
            rows += run_extra_arm(width, cfg, device, outdir, targets, args.extra_arm)
        pd.DataFrame(rows).to_csv(
            os.path.join(outdir, f"intervention_log_{args.extra_arm}.csv"), index=False)
        # runs.csv に追加アームの行を追記 (解析の join 用)
        rcsv = os.path.join(outdir, "runs.csv")
        old = pd.read_csv(rcsv)
        extra = pd.DataFrame([dict(r, arm=args.extra_arm,
                                   run_id=f"{r['run_id']}_{args.extra_arm}")
                              for r in build_runs(cfg)])
        pd.concat([old, extra[~extra.run_id.isin(old.run_id)]],
                  ignore_index=True).to_csv(rcsv, index=False)
        print("EXTRA ARM DONE", flush=True)
        return

    # runs.csv は CLI の --widths に依らず config 全幅分を書く (部分実行でも表は完全)
    all_runs = [dict(r, arm=arm, run_id=f"{r['run_id']}_{arm}")
                for r in build_runs(cfg)
                for arm in cfg["rank_int"]["arms"]]
    with open(os.path.join(outdir, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_runs[0].keys()))
        w.writeheader()
        w.writerows(all_runs)
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)

    t_start = time.time()
    ilog, sanity = [], []
    for width in widths:
        rows, slog = run_width(width, cfg, device, outdir, targets, smoke=args.smoke)
        ilog += rows
        sanity.append(slog)
    pd.DataFrame(ilog).to_csv(os.path.join(outdir, "intervention_log.csv"), index=False)
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(dict(elapsed_sec=round(time.time() - t_start, 1),
                       sanity=sanity, device=device,
                       date=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1,
                  default=str)
    print("sanity:", sanity, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

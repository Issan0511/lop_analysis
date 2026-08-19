"""posreset_0819 Phase 1: 同方向・小ノルムリセット判別のランナー (spec_posreset_0819 §3–7)。

  python -m src.posreset --config configs/posreset_0819.yaml [--regimes A B]
                         [--arms none posonly dironly full] [--smoke] [--device cpu]

レジームごとの流れ (rank_int_0814 の構造を踏襲):
  1. 連続 run 0→t_int+post (= トランク兼参照)。t_int で完全再開スナップショットを保存
  2. スナップショットを読み、固定 eval バッチ上の p̂_i から treated = {i : p̂_i < 0.05} を決定
     (§3.3。lop_metrics.compute_lop_metrics の open_frac / neg_gate_frac と同一定義)
  3. t=0 と同一分布から fresh draw g_i をレジームにつき 1 回だけ引き (§3.4)、
     posonly / dironly / full の 3 アームで再利用。介入の数値計算は float64
     (rank_int_0814 の前例。S3 の 1e-12 判定を float32 丸めで壊さないため)
  4. S3: 介入の数値保証を float64 で判定し、学習再開用の float32 丸め後の値も併記 (§7, §12)
  5. アーム実行。**none を先頭に固定**し、直後に S2 (bit 一致 resume) を判定する。
     S2 が落ちたら介入アームに進まず中止 (rank_int の S1 前例)
  6. intervention_log.csv / unit_traj_*.npz / meta.json を出力
     (runs.csv / verdict.csv / summary.md / figures は解析モジュール側の担当)

**判定は clean eval_loss のみ。dead_frac は PASS/FAIL に使わない** (§9)。
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from .common import (ROOT, load_config, pick_device, build_runs, group_runs,
                     group_name, resolve_outdir)
from .envs import kaiming_mlp_params
from .rank_int import arm_runs, compare_logs
from .train import setup_group, train_group, load_resume, eval_batch
from .lop_metrics import compute_lop_metrics

# fresh draw g_i の generator シード: reset_seed_base + レジーム固有の決定的 offset。
# レジーム間で同じ乱数列を使わないためだけの固定値 (§3.4 は seed 系列の共有のみ要求)。
REGIME_SEED_OFFSET = {"A": 0, "B": 1000}

RESET_ARMS = ("posonly", "dironly", "full")


# ---------------------------------------------------------------- 小道具

def _sha(t):
    """テンソルの生バイト列の sha256 (bit 不変性チェック用)。"""
    return hashlib.sha256(
        np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def _mask_hash(mask_np):
    """treated ブールマスク [h] の sha256 (np.packbits したバイト列に対して)。"""
    return hashlib.sha256(np.packbits(mask_np).tobytes()).hexdigest()


def _max0(t):
    """空テンソルの max を 0.0 に落とす (treated 0 個の seed は検査が空虚に PASS)。"""
    return float(t.max()) if t.numel() else 0.0


def eval_inputs(st):
    """固定 eval バッチの学習器入力 x_in [N,R,d] と教師出力 y [N,R]。
    enc=std のレジームでは centered=False なので x_in は x_raw と一致するが、
    train_group 内の計測と同じ式にしておく。"""
    x_ev, y_ev = eval_batch(st)
    cmask = st["centered"][:, None].float()
    return x_ev - cmask[None] * st["running_mean"][None], y_ev


def current_mu_A(st):
    """レジーム A の解析 µ = E[x] = concat(flip_state, 0.5·1) [R,d]。
    train.make_wdir_ctx の condA 分岐と同一式 (タスク境界で flip_state が変わるので
    呼び出し時点の「現在の µ」になる) [posreset_0819 §5]。"""
    f = st["env"].f
    return torch.cat([st["env"].flip_state,
                      0.5 * torch.ones(st["R"], st["d"] - f, device=st["device"])], dim=1)


# ---------------------------------------------------------------- treated 集合 (§3.3)

def treated_and_pre_metrics(st, cfg, tau):
    """t_int 時点の固定 eval バッチ上で p̂_i = mean(1{w_iᵀx + b_i > 0}) [R,h] を計算し、
    treated = {i : p̂_i < tau} を返す [posreset_0819 §3.3]。

    p̂ は lop_metrics.compute_lop_metrics の open_frac = (pre > 0).float().mean(dim=0)
    と同一定義であり、tau=0.05 = 1 - dead_tau(0.95) なので
    neg_gate_frac = (open_frac < 1 - dead_tau).float().mean(dim=1) が数える集合と厳密に
    一致する (neg_gate_frac は treated_frac そのもの)。"""
    x_in, y_ev = eval_inputs(st)
    m = compute_lop_metrics(st["net"], x_in, y_ev, cfg)
    with torch.no_grad():
        pre, _, _ = st["net"].forward_batch(x_in)          # [N,R,h]
        p_hat = (pre > 0).float().mean(dim=0)              # [R,h]
    treated = p_hat < tau
    return treated, p_hat, dict(dead_frac=m["dead_frac"].cpu().numpy(),
                                eval_loss=m["eval_loss"].cpu().numpy(),
                                neg_gate_frac=m["neg_gate_frac"].cpu().numpy())


# ---------------------------------------------------------------- 介入 (float64, §3.4)

def fresh_draws(regime, R, h, d, reset_seed_base, device):
    """t=0 と同一分布からの fresh draw g_i [R,h,d]。
    envs.kaiming_mlp_params の入力層 W と同一 = 成分独立の U(−√(6/d), +√(6/d))。
    レジームにつき 1 回だけ引き、3 つの reset アームで再利用する [§3.4]。"""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(reset_seed_base) + REGIME_SEED_OFFSET[regime])
    return kaiming_mlp_params(R, h, d, gen, device)[0]


def build_arm_params(net, G32, treated, norm_guard):
    """3 つの reset アームのパラメータを float64 で構成 [§3.4 の表]。

      posonly: w ← ‖g‖·(w/‖w‖) , b ← 0        , v ← 0
      dironly: w ← ‖w‖·(g/‖g‖) , b は**保持**  , v ← 0
      full   : w ← g            , b ← 0        , v ← 0

    ガード: ‖w_i‖ < norm_guard は方向が定義できないので posonly を full にフォールバック。
    treated 外のユニットと c は全アームで一切触らない。

    返り値: (arms64, guard) — arms64[arm] = (W64, b64, v64)、guard [R,h] bool。"""
    W = net.W.double()
    b = net.b.double()
    v = net.v.double()
    G = G32.double()
    wn = W.norm(dim=2)                                    # [R,h]
    gn = G.norm(dim=2)                                    # [R,h]
    assert bool((gn > 0).all()), "fresh draw g_i にゼロノルムのユニットが出た"

    guard = treated & (wn < float(norm_guard))
    tm = treated[:, :, None]
    gm = guard[:, :, None]
    zb = torch.zeros_like(b)
    v_new = torch.where(treated, torch.zeros_like(v), v)

    W_pos = torch.where(tm, (gn / wn.clamp_min(1e-300))[:, :, None] * W, W)
    W_pos = torch.where(gm, G, W_pos)                     # ガード発動ユニットは full と同じ
    W_dir = torch.where(tm, (wn / gn)[:, :, None] * G, W)
    W_ful = torch.where(tm, G, W)

    return dict(posonly=(W_pos, torch.where(treated, zb, b), v_new),
                dironly=(W_dir, b.clone(), v_new),
                full=(W_ful, torch.where(treated, zb, b), v_new)), guard


# ---------------------------------------------------------------- S3 (§7)

def s3_row(i, net, G32, arms64, arms32, treated, guard, tol, c_ref):
    """seed i の S3 検査 [posreset_0819 §7]。

    数値保証 (cos / ノルム) は float64 の介入結果で判定し (tol=s3_tol_f64)、
    学習再開用に float32 へ丸めた後の同じ量を _f32 列に併記する (float32 の
    eps≈1.2e-7 に律速されるので**判定はしない**、§12)。float32 側も参照値
    (‖g‖, ‖w_pre‖) は float32 原値の厳密な upcast を使うので、差は丸め誤差のみ。

    **ガード発動ユニット (posonly が full にフォールバックした分) は posonly の
    cos / ノルム検査から除外する** (定義上 posonly の規約を満たさないため)。代わりに
    s3_guard_full_exact_* で「フォールバック先が確かに full と同一 (w=g)」を検査する。

    §7 の字義は dironly について ‖w_post‖=‖w_pre‖ しか要求しないが、それだけでは
    「dironly が full と同じ fresh draw g_i を共有している」(§3.4 の
    「g_i はユニットごとに1回だけ生成して3アームで再利用する」) を検出できない。
    P3 の科学的意味はこの共有に依存するので s3_dironly_cos_g_* を追加して判定に入れる。"""
    t = treated[i]
    g = guard[i]
    pos_m = t & ~g                       # 真に posonly 規約が要求されるユニット
    W_pre64 = net.W[i].double()
    G64 = G32[i].double()
    out = {}

    for tag, Warm in (("f64", {a: arms64[a][0][i] for a in RESET_ARMS}),
                      ("f32", {a: arms32[a][0][i].double() for a in RESET_ARMS})):
        wp, wq = W_pre64[pos_m], Warm["posonly"][pos_m]
        cos = (wp * wq).sum(-1) / (wp.norm(dim=-1) * wq.norm(dim=-1)).clamp_min(1e-300)
        out[f"s3_posonly_cos_err_{tag}"] = _max0((cos - 1.0).abs())
        gnp = G64[pos_m].norm(dim=-1)
        out[f"s3_posonly_norm_relerr_{tag}"] = _max0(
            (wq.norm(dim=-1) - gnp).abs() / gnp.clamp_min(1e-300))
        wpre_t, wdir_t = W_pre64[t], Warm["dironly"][t]
        out[f"s3_dironly_norm_relerr_{tag}"] = _max0(
            (wdir_t.norm(dim=-1) - wpre_t.norm(dim=-1)).abs()
            / wpre_t.norm(dim=-1).clamp_min(1e-300))
        # dironly の方向が full と同じ fresh draw g_i であること (§3.4 の g 再利用)。
        # ガード発動ユニット (‖w_pre‖≈0) は dironly が ‖w‖·ĝ = 0 になり方向が定義でき
        # ないので、posonly と同じく pos_m で除外する (ノルム検査は 0=0 で通るため全
        # treated のまま)。
        wdir_p, gp = Warm["dironly"][pos_m], G64[pos_m]
        cos_g = (wdir_p * gp).sum(-1) / (wdir_p.norm(dim=-1) * gp.norm(dim=-1)).clamp_min(1e-300)
        out[f"s3_dironly_cos_g_{tag}"] = _max0((cos_g - 1.0).abs())
        # full: w_post ≡ g (float64 では厳密 0、float32 でも代入なので厳密 0 のはず)
        out[f"s3_full_exact_{tag}"] = _max0((Warm["full"][t] - G64[t]).abs())
        # ガード発動ユニットの posonly は full 規約 (w=g) にフォールバックしていること
        out[f"s3_guard_full_exact_{tag}"] = _max0((Warm["posonly"][g] - G64[g]).abs())

    # 以下は「実際に resume に載る float32 の状態」に対する厳密検査
    ok_v, ok_b, ok_hash = True, True, True
    pre_untouched = (_sha(net.W[i][~t]), _sha(net.b[i][~t]), _sha(net.v[i][~t]))
    for a in RESET_ARMS:
        Wa, ba, va = (x[i] for x in arms32[a])
        ok_v &= bool((va[t] == 0).all())
        ok_b &= bool((ba[t] == 0).all()) if a in ("posonly", "full") \
            else bool((ba[t] == net.b[i][t]).all())
        ok_hash &= (_sha(Wa[~t]), _sha(ba[~t]), _sha(va[~t])) == pre_untouched
    # 出力バイアス c は介入対象外 (アーム側は snap を deepcopy して W/b/v だけ差し替える)。
    # net.c 自体が in-place で汚れていないことを、スナップショット由来の独立オブジェクト
    # c_ref と突き合わせて確認する (自分自身との比較は検出力ゼロなので使わない)。
    ok_c = _sha(net.c[i]) == _sha(c_ref[i])
    out["s3_readout_zero_ok"] = ok_v
    out["s3_bias_ok"] = ok_b
    out["s3_untreated_hash_ok"] = bool(ok_hash and ok_c)
    out["s3_pass"] = bool(
        ok_v and ok_b and ok_hash and ok_c
        and out["s3_posonly_cos_err_f64"] < tol
        and out["s3_posonly_norm_relerr_f64"] < tol
        and out["s3_dironly_norm_relerr_f64"] < tol
        and out["s3_dironly_cos_g_f64"] < tol
        and out["s3_full_exact_f64"] == 0.0
        and out["s3_guard_full_exact_f64"] == 0.0)
    return out


# ---------------------------------------------------------------- S2 (§7)

# resume 境界行 (step == t_int) で構造的に NaN になる列 [lop_metrics.compute_b_metrics]。
# どちらも bctx が持ち回る「直前の lop step の dead / p=0 マスク」との比較なので、
# 直前の lop step を持たない resume 側の先頭行では定義できない (軌道の不一致ではない)。
S2_RESUME_NA_COLS = ("dead_persist_frac", "p_zero_persist_frac")


def _s2_stage(outdir, gnames, dst, drop_cols):
    """compare_logs に渡す CSV 一式を dst に用意する (results/ を汚さないため一時領域)。

    3 つの前処理を入れる。いずれも**両アームに対称に**適用するので比較の意味は変わらない:
      (i)  本 config は coupling キーを持たず postswitch_err_*.csv が存在しないが、
           compare_logs は無条件に読むのでヘッダのみのスタブを置く
      (ii) pandas の (da != db) は NaN != NaN を True にするため、CSV 上の "nan" を
           非 NA のセンチネル文字列に置換する (NaN 同士を一致扱いにする)。b_metrics の
           b_mean_alive / beta_p10 等は全ユニット dead の step で NaN になり得る
      (iii) drop_cols を落とす (S2_RESUME_NA_COLS 用)"""
    os.makedirs(dst, exist_ok=True)
    for g in gnames:
        for prefix in ("lop_metrics", "online_loss"):
            d = pd.read_csv(os.path.join(outdir, f"{prefix}_{g}.csv"),
                            dtype=str, keep_default_na=False)
            d = d.replace({"nan": "_NA_", "-nan": "_NA_", "": "_NA_"})
            d = d.drop(columns=[c for c in drop_cols if c in d.columns])
            d.to_csv(os.path.join(dst, f"{prefix}_{g}.csv"), index=False)
        src = os.path.join(outdir, f"postswitch_err_{g}.csv")
        out = os.path.join(dst, f"postswitch_err_{g}.csv")
        if os.path.exists(src):
            shutil.copyfile(src, out)
        else:
            with open(out, "w") as fh:
                fh.write("switch_step,run_id,post_err\n")


def s2_compare(outdir, gname_a, gname_b, t_int):
    """S2 (§7): rank_int.compare_logs を**書き換えずに**再利用して bit 一致 resume を判定。

    2 段に分けて呼ぶ:
      post     : t_min = t_int+1 で step > t_int の全行・**全列**を厳密比較
                 (online_loss は loss_bin 境界にしか行が無いので t_int と同じ集合)
      boundary : t_min = t_int で境界行 (step == t_int、resume 直後の状態そのもの) を
                 含めて比較。ただし S2_RESUME_NA_COLS は定義上 NaN なので除外する
    両者の diff の和が空なら PASS。除外列は軌道を担う量ではない (persist 系のみ)。"""
    with tempfile.TemporaryDirectory() as td:
        post, bnd = os.path.join(td, "post"), os.path.join(td, "bnd")
        _s2_stage(outdir, (gname_a, gname_b), post, ())
        _s2_stage(outdir, (gname_a, gname_b), bnd, S2_RESUME_NA_COLS)
        diffs = {f"post_{k}": v for k, v in
                 compare_logs(post, gname_a, gname_b, t_int + 1).items()}
        diffs.update({f"boundary_{k}": v for k, v in
                      compare_logs(bnd, gname_a, gname_b, t_int).items()})
        return diffs


# ---------------------------------------------------------------- 副次時系列 probe (§5)

def make_probe(treated, acc):
    """treated ユニットの p̂ / ‖w‖ / β / cos(w,µ) を probe_steps ごとに acc へ蓄積する
    [posreset_0819 §5]。train_group から probe(st, step) として呼ばれる。

    eval_batch は乱数を消費しないので (train.train_group の docstring 参照)、probe を
    付けても学習軌道は変わらない。"""
    idx = [torch.nonzero(treated[i]).squeeze(1) for i in range(treated.shape[0])]

    def probe(st, step):
        with torch.no_grad():
            x_in, _ = eval_inputs(st)
            net = st["net"]
            pre, _, _ = net.forward_batch(x_in)                 # [N,R,h]
            p_hat = (pre > 0).float().mean(dim=0)               # [R,h]
            wn = net.W.norm(dim=2)                              # [R,h]
            nanv = torch.full_like(wn, float("nan"))
            if st["exp"] == "B":
                env = st["env"]
                # µ=0 かつ Σ=I (c=0, kappa=1) なら β_i = b_i/‖w_i‖ が厳密。
                # そうでない config で流用されたときは一般式へフォールバックする。
                if env.sk is None and bool((env.mu == 0).all()):
                    beta = net.b / wn.clamp_min(1e-30)
                else:
                    wSw = (net.W ** 2).sum(2)
                    if env.sk is not None:
                        wu = torch.einsum("rhd,d->rh", net.W, env.u)
                        wSw = wSw + ((env.sk + 1.0) ** 2 - 1.0)[:, None] * wu ** 2
                    wmu = torch.einsum("rhd,rd->rh", net.W, env.mu)
                    beta = (wmu + net.b) / wSw.clamp_min(1e-30).sqrt()
                cos_u_mu = nanv
            else:
                beta = nanv
                mu = current_mu_A(st)                           # [R,d]
                wmu = torch.einsum("rhd,rd->rh", net.W, mu)
                # 符号つき cos (ゲート開放は wᵀµ>0 半空間なので符号を落とさない、§5)
                cos_u_mu = wmu / (wn * mu.norm(dim=1)[:, None]).clamp_min(1e-30)
            for i, ii in enumerate(idx):
                a = acc[i]
                a["steps"].append(int(step))
                for k, src in (("p_hat", p_hat), ("w_norm", wn),
                               ("beta", beta), ("cos_u_mu", cos_u_mu)):
                    a[k].append(src[i, ii].detach().cpu().numpy().astype(np.float32))
    return probe


def new_acc(R):
    return [dict(steps=[], p_hat=[], w_norm=[], beta=[], cos_u_mu=[]) for _ in range(R)]


def write_traj(outdir, regime, arm, base_runs, treated, acc, t_int, thash):
    """unit_traj_{regime}_{seed}_{arm}.npz を書く [posreset_0819 §5]。
    steps [T] / unit_idx [U] / p_hat・w_norm・beta・cos_u_mu [T,U]。"""
    for i, r in enumerate(base_runs):
        ii = torch.nonzero(treated[i]).squeeze(1).cpu().numpy().astype(np.int64)
        a = acc[i]
        T, U = len(a["steps"]), len(ii)
        arrs = {k: np.asarray(a[k], dtype=np.float32).reshape(T, U)
                for k in ("p_hat", "w_norm", "beta", "cos_u_mu")}
        np.savez(os.path.join(outdir, f"unit_traj_{regime}_{r['seed']}_{arm}.npz"),
                 steps=np.asarray(a["steps"], dtype=np.int64), unit_idx=ii,
                 regime=np.str_(regime), seed=np.int64(r["seed"]), arm=np.str_(arm),
                 t_int=np.int64(t_int), treated_hash=np.str_(thash[i]), **arrs)


# ---------------------------------------------------------------- レジーム 1 本

def run_regime(gkey, base_runs, cfg, device, outdir, arms, reuse_snapshot=False):
    P = cfg["posreset"]
    t_int, post = int(P["t_int"]), int(P["post_steps"])
    total = t_int + post
    regime = gkey[0]
    gbase = group_name(gkey)                       # A_w100 / B_w20
    gcont = f"{gbase}_cont"
    log = dict(regime=regime, gbase=gbase, t_int=t_int, total=total)

    # --- 1. 連続 run (トランク兼参照) + t_int スナップショット
    snap_path = os.path.join(outdir, "snapshots", f"{gcont}_step{t_int}.pt")
    if reuse_snapshot and os.path.exists(snap_path):
        print(f"=== [{gbase}] reuse snapshot {snap_path}", flush=True)
    else:
        print(f"=== [{gbase}] continuous trunk 0->{total}", flush=True)
        _, el = train_group(gkey, arm_runs(base_runs, "cont"), cfg, device, outdir,
                            total_steps=total, ckpts=[], gname=gcont,
                            snapshot_steps=[t_int])
        print(f"    done {el:.1f}s ({total/max(el,1e-9):.0f} steps/s)", flush=True)
    snap = torch.load(snap_path, weights_only=False)
    log["snapshot_sha256"] = {k: _sha(v) for k, v in snap["net"].items()}
    log["snapshot_sha256"]["running_mean"] = _sha(snap["running_mean"])

    # --- 2. treated 集合 (§3.3)
    st_tmp = setup_group(gkey, base_runs, cfg, device)
    load_resume(st_tmp, snap)
    treated, _p_hat, pre_m = treated_and_pre_metrics(st_tmp, cfg, float(P["p_hat_tau"]))
    net = st_tmp["net"]
    R, h = treated.shape
    tfrac = treated.float().mean(dim=1).cpu().numpy()
    tnp = treated.cpu().numpy()
    thash = [_mask_hash(tnp[i]) for i in range(R)]
    # treated の定義が lop_metrics の neg_gate_frac (= open_frac < 1-dead_tau) と
    # 厳密に同じ集合を数えていることの実測確認 [§3.3]。neg_gate_frac の閾値は
    # 1-dead_tau に固定なので、tau をそれと違う値にしたとき (スモークのみ) は
    # 照合できないのでスキップする。発散 run は lop_metrics 側が NaN を返すので
    # 比較から外す (発散自体は log に残す)。
    tau = float(P["p_hat_tau"])
    ng = pre_m["neg_gate_frac"]
    fin = np.isfinite(ng)
    log["n_nonfinite_at_tint"] = int((~fin).sum())
    if abs(tau - (1.0 - float(cfg["common"]["dead_tau"]))) < 1e-12:
        assert np.allclose(tfrac[fin], ng[fin], atol=1e-6), \
            "treated_frac が lop_metrics の neg_gate_frac と一致しない (定義ずれ)"
        log["treated_eq_neg_gate"] = True
    else:
        log["treated_eq_neg_gate"] = f"skipped (p_hat_tau={tau} != 1-dead_tau)"
    print(f"    treated_frac: {np.round(tfrac, 3).tolist()}", flush=True)

    # --- 3. fresh draw と介入 (float64, §3.4)
    G32 = fresh_draws(regime, R, h, st_tmp["d"], P["reset_seed_base"], device)
    arms64, guard = build_arm_params(net, G32, treated, P["norm_guard"])
    arms32 = {a: tuple(x.float() for x in v) for a, v in arms64.items()}

    # --- 4. S3 (§7)
    ilog = []
    for i, r in enumerate(base_runs):
        row = dict(regime=regime, exp=r["exp"], width=r["width"], seed=r["seed"],
                   base_run_id=r["run_id"], t_int=t_int, h=h,
                   n_treated=int(tnp[i].sum()), treated_frac=float(tfrac[i]),
                   n_guard_fallback=int(guard[i].sum()), treated_hash=thash[i],
                   pre_dead_frac=float(pre_m["dead_frac"][i]),
                   pre_eval_loss=float(pre_m["eval_loss"][i]))
        row.update(s3_row(i, net, G32, arms64, arms32, treated, guard,
                          float(P["s3_tol_f64"]), snap["net"]["c"]))
        ilog.append(row)
    log["S3"] = "PASS" if all(r["s3_pass"] for r in ilog) else "FAIL"
    log["s3_worst_f64"] = {k: max(r[k] for r in ilog) for k in
                           ("s3_posonly_cos_err_f64", "s3_posonly_norm_relerr_f64",
                            "s3_dironly_norm_relerr_f64", "s3_dironly_cos_g_f64",
                            "s3_full_exact_f64", "s3_guard_full_exact_f64")}
    log["s3_worst_f32"] = {k: max(r[k] for r in ilog) for k in
                           ("s3_posonly_cos_err_f32", "s3_posonly_norm_relerr_f32",
                            "s3_dironly_norm_relerr_f32", "s3_dironly_cos_g_f32",
                            "s3_full_exact_f32", "s3_guard_full_exact_f32")}
    log["n_guard_fallback_total"] = int(guard.sum())
    print(f"    S3: {log['S3']}  worst_f64={log['s3_worst_f64']}", flush=True)

    # --- 5. アーム実行 (none を先頭に固定 -> 直後に S2 を判定)
    ordered = [a for a in P["arms"] if a in arms]
    if "none" in ordered:
        ordered = ["none"] + [a for a in ordered if a != "none"]
    probe_steps = list(range(t_int, total + 1, int(P["probe_every"])))
    log["S2"] = "SKIP (none arm not requested)"
    if "none" not in ordered:
        print("    !! WARNING: none アームを外したので S2 (bit 一致 resume) を判定できない。"
              "本走では必ず none を含めること", flush=True)
    for arm in ordered:
        snap_arm = copy.deepcopy(snap)
        if arm != "none":
            Wa, ba, va = arms32[arm]
            snap_arm["net"]["W"] = Wa.clone()
            snap_arm["net"]["b"] = ba.clone()
            snap_arm["net"]["v"] = va.clone()
        acc = new_acc(R)
        gname = f"{gbase}_{arm}"
        print(f"=== [{gbase}] arm {arm} {t_int}->{total}", flush=True)
        _, el = train_group(gkey, arm_runs(base_runs, arm), cfg, device, outdir,
                            total_steps=total, ckpts=[], gname=gname,
                            start_step=t_int, resume_state=snap_arm,
                            probe=make_probe(treated, acc), probe_steps=probe_steps)
        write_traj(outdir, regime, arm, base_runs, treated, acc, t_int, thash)
        print(f"    done {el:.1f}s", flush=True)

        if arm == "none":
            diffs = s2_compare(outdir, gcont, gname, t_int)
            log["S2"] = "PASS" if not diffs else f"FAIL: {diffs}"
            log["S2_note"] = ("step>t_int は全列厳密一致。境界行 (step==t_int) のみ "
                              f"{list(S2_RESUME_NA_COLS)} を除外 (resume 側は直前の "
                              "lop step を持たず定義上 NaN)")
            print(f"    S2: {log['S2']}", flush=True)
            if diffs:
                raise SystemExit(f"S2 FAILED ({gbase}): {diffs} — 介入アームを中止")

    # --- 6. 適格性の記録 (§3.3。除外はしない、層別報告用の記録のみ)
    n_ok = int((tfrac >= float(P["treated_frac_min"])).sum())
    log["treated_frac"] = dict(values=[round(float(x), 4) for x in tfrac],
                               mean=round(float(tfrac.mean()), 4),
                               min=round(float(tfrac.min()), 4),
                               n_ge_min=n_ok, n_seeds=int(R),
                               eligible=bool(n_ok >= int(P["treated_frac_min_seeds"])))
    return ilog, log


# ---------------------------------------------------------------- CLI

def apply_smoke(cfg):
    """--smoke: 数十秒で終わる縮約 (幅 5、seed 2 本、t_int 4k / post 6k)。

    p_hat_tau だけは **スモーク専用** に 0.5 へ緩める。本番の treated 集合は µ経路 /
    b経路の dead が育った 500k 時点で初めて非空になるので、4k step では 0.05 基準だと
    treated が空 (特にレジーム B は完全に 0) になり、介入の数値・S3・npz が一切
    実行されない。介入コードは「なぜ treated になったか」に依存しないため、tau を
    上げるだけで全経路を検証できる。**config 側の凍結値 0.05 (§3.3) は不変**。"""
    cfg = copy.deepcopy(cfg)
    cfg["common"].update(seeds=[0, 1], eval_batch=200, lop_every=500, loss_bin=1000)
    cfg["condA"]["widths"] = [5]
    cfg["condB"]["widths"] = [5]
    cfg["posreset"].update(t_int=4000, post_steps=6000, probe_every=1000,
                           p_hat_tau=0.5)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/posreset_0819.yaml")
    ap.add_argument("--regimes", nargs="*", default=None, choices=["A", "B"])
    ap.add_argument("--arms", nargs="*", default=None,
                    help="実行するアーム (既定は config の 4 本。none は常に先頭に回す)")
    ap.add_argument("--smoke", action="store_true",
                    help="t_int=4k / post=6k / width 5 / seed 2 本 (results/_smoke_posreset)")
    ap.add_argument("--reuse-snapshot", action="store_true",
                    help="既存スナップショットがあればトランクを再実行しない (部分再走用)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.smoke:
        cfg = apply_smoke(cfg)
        outdir = resolve_outdir(args.config,
                                outdir=os.path.join(ROOT, "results", "_smoke_posreset"))
    else:
        outdir = resolve_outdir(args.config)
    # トランク・参照 run の長さは posreset の t_int + post_steps で一意に決める
    cfg["common"]["total_steps"] = int(cfg["posreset"]["t_int"]) + \
        int(cfg["posreset"]["post_steps"])
    os.makedirs(outdir, exist_ok=True)
    print(f"outdir: {outdir}", flush=True)

    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)

    arms = args.arms or list(cfg["posreset"]["arms"])
    groups = group_runs(build_runs(cfg))
    want = args.regimes or ["A", "B"]

    t0 = time.time()
    ilog, sanity = [], []
    for gkey, gruns in sorted(groups.items(), key=lambda kv: kv[0][0]):
        if gkey[0] not in want:
            continue
        rows, slog = run_regime(gkey, gruns, cfg, device, outdir, arms,
                                reuse_snapshot=args.reuse_snapshot)
        ilog += rows
        sanity.append(slog)
    pd.DataFrame(ilog).to_csv(os.path.join(outdir, "intervention_log.csv"), index=False)
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(dict(elapsed_sec=round(time.time() - t0, 1), device=device,
                       date=time.strftime("%Y-%m-%d %H:%M:%S"),
                       # S1 の証拠 [posreset_0819 §7]。スレッド数は実測で lop_metrics の
                       # eff_rank を %.6g の最下位桁で動かす (LAPACK の縮約順序) ため、
                       # cont と none アームが違うスレッド数で走ると S2 が誤 FAIL する。
                       # summary.md の S1 行 (§10) はこの 2 キーを読む。
                       omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                       torch_num_threads=torch.get_num_threads(),
                       arms=arms, smoke=bool(args.smoke), sanity=sanity),
                  fh, indent=1, default=str)
    for s in sanity:
        print(f"[{s['regime']}] S2={s['S2']} S3={s['S3']} "
              f"treated_frac={s['treated_frac']}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

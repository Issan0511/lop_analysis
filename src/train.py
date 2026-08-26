"""グループ ((exp, width) 単位) のベクトル化オンライン学習ループ。

各系列は batch=1 の plain SGD ([D][J])。系列 (条件×シード) を R 次元に平坦化して並列化
(仕様書 §8)。中心化 (enc=centered) は学習器入力の前処理であり、教師は生入力を見る。
"""
import os
import time
import csv
import torch

from .common import group_name, switch_steps
from .envs import SCREnv, LTUTarget, GaussEnv, MLPTeacher, kaiming_mlp_params
from .nets import VecMLP
from .lop_metrics import compute_lop_metrics

SEED_BASE = {"A": 10000, "B": 20000}


def make_gens(exp, width, device, offset=0):
    """入力・教師・初期化・eval・method (S&P ノイズ / CBP 再サンプル) で generator を分離
    (仕様書 §8)。既存 4 本のシードは不変 (method 追加は末尾)。"""
    base = SEED_BASE[exp] + width + offset
    gens = {}
    # 既存 5 本のシードは不変 (noise は末尾に追加 [bias_margin_0814 §2.1])
    for i, name in enumerate(["init", "input", "teacher", "eval", "method", "noise"]):
        g = torch.Generator(device=device)
        g.manual_seed(base + 100 * (i + 1))
        gens[name] = g
    return gens


def setup_group(gkey, runs, cfg, device):
    exp, width, batch = gkey[0], gkey[1], gkey[2]
    R = len(runs)
    # generator_offset は独立確認走の乱数系列を seed 表示軸と分離して
    # 切り替えるためのオフセット [function_blind_direct_0823] 。未指定=0 は
    # 従来の make_gens(exp, width, device) と乱数消費も含めて同一。
    gens = make_gens(exp, width, device, offset=int(cfg["common"].get("generator_offset", 0)))
    A, B = cfg["condA"], cfg["condB"]

    period = torch.tensor([r["period"] for r in runs], dtype=torch.long)  # CPU
    lr = torch.tensor([r["lr"] for r in runs], device=device)
    centered = torch.tensor([r["enc"] == "centered" for r in runs], device=device)

    if exp == "A":
        d = A["m"]
        env = SCREnv(R, A["m"], A["f"], period, gens["input"], device)
        # target_out_scale は教師出力全体に掛ける定数 [teachw_0820 §3]。未指定 = 1.0 で
        # 既存 config は bit 一致 (LTUTarget 側で恒等分岐)。
        teacher = LTUTarget(R, A["m"], A["target_hidden"], A["beta"], gens["teacher"],
                            device, out_scale=float(A.get("target_out_scale", 1.0) or 1.0))
    else:
        d = B["d"]
        cvals = [r["c"] for r in runs]
        kvals = [r.get("kappa", 1) for r in runs]
        env = GaussEnv(R, d, cvals, gens["input"], device, kappa=kvals,
                       spike_dir=B.get("spike_dir", "ones"))
        # 教師幅は既定で学習器幅 (従来互換)。target_hidden 指定時のみ分離し、
        # 条件A と同じ容量ギャップを条件B にも作る [center_selfcov_0814 §2.1]。
        th = B.get("target_hidden") or width
        teacher = MLPTeacher(R, th, d, period, gens["teacher"], device)

    # batch: 1=オンライン SGD, 整数=iid ミニバッチ平均, "full"=フルバッチ GD
    #   (A: 全サポート厳密列挙 / B: full_batch_B サンプル近似)
    batch_n = None
    if batch != 1 and batch != "full":
        batch_n = int(batch)
    elif batch == "full" and exp == "B":
        batch_n = int(B.get("full_batch_B", 1024))

    # 介入手法 [methods_sde_0813]: グループ内で単一 (group_runs のキーに method を含む)
    mcfg = runs[0].get("method_cfg", {"name": "none"})
    if mcfg["name"] != "none" and batch != 1:
        raise NotImplementedError("methods は batch=1 (標準 SGD) のみ対応")
    act_alpha = float(mcfg.get("alpha", 0.0)) if mcfg["name"] == "leaky" else 0.0

    cond = A if exp == "A" else B
    net = VecMLP(R, width, d, gens["init"], device, act_alpha=act_alpha,
                 freeze_bias=bool(cond.get("freeze_bias", False)))
    running_mean = torch.zeros(R, d, device=device)

    cbp = None
    if mcfg["name"] == "cbp":
        cbp = dict(util=torch.zeros(R, width, device=device),
                   age=torch.zeros(R, width, device=device),
                   acc=0.0,   # 置換数の端数アキュムレータ (rho, h はグループ内で共通)
                   n_reset=torch.zeros(R, device=device))  # 累積 reset 数 [cbp_harm_0815 S3]

    # LoP 計測用固定バッチ素材 (eval generator)
    N = cfg["common"]["eval_batch"]
    if exp == "A":
        eval_fixed = torch.randint(0, 2, (N, A["m"] - A["f"]),
                                   generator=gens["eval"], device=device).float()
    else:
        eval_fixed = torch.randn(N, d, generator=gens["eval"], device=device)

    return dict(exp=exp, width=width, batch=batch, batch_n=batch_n, R=R, d=d,
                env=env, teacher=teacher, net=net,
                running_mean=running_mean, lr=lr, centered=centered, period=period,
                eval_fixed=eval_fixed, runs=runs, device=device,
                alpha=A["center_alpha"], gname=group_name(gkey),
                method=mcfg, gen_method=gens["method"], cbp=cbp, gens=gens,
                # 教師出力への加法ノイズ [bias_margin_0814 §2.1]。学習ループ専用で、
                # eval_batch() には決して適用しない (clean teacher 基準を保つ)。
                noise_sd=float(cond.get("target_noise_sd", 0.0) or 0.0))


def apply_method(st, a):
    """介入フック: sgd_step 直後に呼ぶ (batch=1 のみ)。a は当ステップの活性 [R,h]。

    - snp: 毎ステップ w <- (1-shrink) w + perturb*zeta (W, b, v のみ、c は対象外)。
      perturb は初期化スケールに正規化しない素の等方ガウス (SDE 上は等方 diffusion 床)。
    - cbp: Dohare 準拠の R 次元ベクトル化。util = decay*util + (1-decay)|v||a|、
      acc += rho*h の floor 分だけ age > maturity の util 最小ユニットを再初期化
      (W kaiming 再サンプル / b=0 / v=0 で関数を壊さない)。端数は繰越。
      eligible 不足時は不足分を切り捨てる (積み残しはしない)。"""
    m, net = st["method"], st["net"]
    name = m["name"]
    if name == "snp":
        shrink, perturb = float(m["shrink"]), float(m["perturb"])
        for p in (net.W, net.b, net.v):
            p.mul_(1.0 - shrink)
            if perturb > 0:
                p.add_(perturb * torch.randn(p.shape, generator=st["gen_method"],
                                             device=st["device"]))
    elif name == "cbp":
        cb = st["cbp"]
        decay = float(m["decay"])
        cb["util"].mul_(decay).add_((1 - decay) * (net.v.abs() * a.abs()))
        cb["age"] += 1
        cb["acc"] += float(m["rho"]) * st["width"]
        n = int(cb["acc"])
        if n > 0:
            cb["acc"] -= n
            eligible = cb["age"] > float(m["maturity"])                  # [R,h]
            util_m = cb["util"].masked_fill(~eligible, float("inf"))
            vals, idx = torch.topk(util_m, min(n, st["width"]), dim=1, largest=False)
            sel = torch.zeros_like(eligible)
            sel.scatter_(1, idx, vals.isfinite())                        # eligible のみ置換
            if sel.any():
                Wn, _, _, _ = kaiming_mlp_params(st["R"], st["width"], st["d"],
                                                 st["gen_method"], st["device"])
                net.W[sel] = Wn[sel]
                net.b[sel] = 0.0
                net.v[sel] = 0.0
                cb["util"][sel] = 0.0
                cb["age"][sel] = 0.0
                cb["n_reset"] += sel.sum(dim=1).float()


def eval_batch(st):
    """現在の環境状態での計測用バッチ (x_raw [N,R,d], y [N,R])。"""
    if st["exp"] == "A":
        N = st["eval_fixed"].shape[0]
        f = st["env"].f
        flip = st["env"].flip_state[None].expand(N, -1, -1)              # [N,R,f]
        rnd = st["eval_fixed"][:, None, :].expand(-1, st["R"], -1)        # [N,R,m-f]
        x = torch.cat([flip, rnd], dim=2)
    else:
        # 異方 Sigma でも eval が入力分布と一致するよう z に Sigma^{1/2} を適用
        x = st["env"].mu[None] + st["env"]._transform(st["eval_fixed"][:, None, :])
    y = st["teacher"](x)
    return x, y


def save_ckpt(st, step, outdir):
    path = os.path.join(outdir, "ckpts", f"{st['gname']}_step{step}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(step=step,
                    net=st["net"].state_dict(),
                    env=st["env"].state_dict(),
                    teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(),
                    runs=st["runs"]), path)


def save_snapshot(st, step, outdir):
    """完全再開スナップショット [rank_int_0814 §3]。save_ckpt との違いは RNG 状態
    (gens) を含むこと。full-batch 決定論下では resume 後の軌道が bit 一致する。"""
    path = os.path.join(outdir, "snapshots", f"{st['gname']}_step{step}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(step=step,
                    net=st["net"].state_dict(),
                    env=st["env"].state_dict(),
                    teacher=st["teacher"].state_dict(),
                    running_mean=st["running_mean"].clone(),
                    gens={k: g.get_state() for k, g in st["gens"].items()},
                    runs=st["runs"]), path)
    return path


def load_resume(st, snap):
    """setup_group 済みの st にスナップショット状態を書き戻す (in-place)。"""
    st["net"].load_state(snap["net"])
    st["env"].load_state(snap["env"])
    st["teacher"].load_state(snap["teacher"])
    st["running_mean"].copy_(snap["running_mean"])
    for k, g in st["gens"].items():
        g.set_state(snap["gens"][k])


def make_bctx(st, cfg):
    """b/β 指標用のコンテキスト [bias_margin_0814 §2.3]。
    cfg["common"]["b_metrics"] が真のときだけ有効 (既定は無効 = 既存 config 互換)。
    条件B は µ, Σ を解析値で渡し、条件A は None にして eval バッチからの経験推定に落とす。"""
    import numpy as np
    from .w_direction import spike_dir_vec

    if not cfg["common"].get("b_metrics"):
        return None
    if st["exp"] == "B":
        return dict(mu=st["env"].mu.detach().cpu().numpy().astype(np.float64),
                    kappa=np.array([float(r.get("kappa", 1)) for r in st["runs"]]),
                    u=spike_dir_vec(cfg["condB"].get("spike_dir", "ones"), st["d"]),
                    prev_dead=None)
    return dict(mu=None, kappa=None, u=None, prev_dead=None)


def make_wdir_ctx(st, cfg):
    """W 方向指標用の解析パラメータ [center_selfcov_0814 §2.2]。
    cfg["common"]["w_dir_metrics"] が真のときだけ有効 (既定は無効 = 既存 config 互換)。

    条件A: Σ = (1/4)I で完全等方 → u=None (Σ 系は NaN)。µ は E[xxᵀ] 側でのみ意味を持つ。
    条件B: u は spike_dir から決定的、κ は run 別、µ = (c/√d)·1。
    サンプル推定はせず解析値を使う。"""
    import numpy as np
    from .w_direction import spike_dir_vec

    if not cfg["common"].get("w_dir_metrics"):
        return None
    R, d = st["R"], st["d"]
    if st["exp"] == "B":
        u = spike_dir_vec(cfg["condB"].get("spike_dir", "ones"), d)
        kappa = np.array([float(r.get("kappa", 1)) for r in st["runs"]])
        mu = st["env"].mu.detach().cpu().numpy().astype(np.float64)
    else:
        # 条件A: 等方 → Σ 系は縮退。µ は現在の flip 状態 + ランダムビットの 1/2
        u = None
        kappa = np.ones(R)
        f = st["env"].f
        mu = np.concatenate(
            [st["env"].flip_state.detach().cpu().numpy().astype(np.float64),
             0.5 * np.ones((R, d - f))], axis=1)
    return dict(u=u, kappa=kappa, mu=mu, prev_e1=None)


def build_lop_steps(cfg, total, period_val):
    """計測ステップ集合: 粗い定期計測 (lop_every) + タスク境界周辺の密な窓
    (cfg["coupling"]: pre_window/post_window/fine_stride、実験(5) methods coupling_ab 用)。
    coupling キーが無い既存 config では従来の (t+1)%lop_every==0 と等価。"""
    C = cfg["common"]
    steps = set(range(0, total + 1, C["lop_every"]))
    cp = cfg.get("coupling") or {}
    pre, post = cp.get("pre_window", 0), cp.get("post_window", 0)
    stride = max(1, cp.get("fine_stride", C["lop_every"]))
    if (pre or post) and period_val:
        for s0 in switch_steps(period_val, total):
            lo, hi = max(0, s0 - pre), min(total, s0 + post)
            steps.update(range(lo, hi + 1, stride))
    steps.add(0)
    steps.add(total)
    return steps


def train_group(gkey, runs, cfg, device, outdir, total_steps=None, ckpts=None,
                start_step=0, resume_state=None, gname=None, snapshot_steps=(),
                probe=None, probe_steps=(), pre_update_probe=None,
                pre_update_probe_steps=()):
    """start_step / resume_state / gname / snapshot_steps は warm-start 用
    [rank_int_0814 §3]。resume_state は save_snapshot が書いた dict (介入アームでは
    呼び出し側が net["W"] を差し替えてから渡す)。gname はログファイル名の上書き
    (同一グループ条件で複数アームを別名保存するため)。

    probe / probe_steps は treated ユニットの副次時系列取得用フック
    [posreset_0819 §5]。probe が None でないとき、絶対 step が probe_steps に含まれる
    時点で probe(st, step) を呼ぶ。snapshot_steps と同じくループ本体先頭で判定するので
    step==start_step (介入直後の状態) も対象になり、末尾 total はループ後に補う。

    probe は st を読むだけで、内部から eval_batch(st) を呼んでよい: eval_batch は
    exp=A なら事前抽選済み st["eval_fixed"] と env.flip_state、exp=B なら env.mu と
    eval_fixed への Sigma^{1/2} 適用しか使わず、**generator を一切消費せず env.t も
    進めない**ため、probe の有無で学習軌道は変わらない (probe=None は厳密な no-op)。"""
    C = cfg["common"]
    total = total_steps or C["total_steps"]
    ckpt_set = set(ckpts if ckpts is not None else C["checkpoints"])
    st = setup_group(gkey, runs, cfg, device)
    if gname:
        st["gname"] = gname
    if resume_state is not None:
        load_resume(st, resume_state)
    net, env, teacher = st["net"], st["env"], st["teacher"]
    alpha, centered, lr = st["alpha"], st["centered"], st["lr"]
    cmask = centered[:, None].float()

    period_val = int(st["period"][0].item()) if len(st["period"]) else 0
    if cfg.get("coupling") and len(set(int(p) for p in st["period"])) > 1:
        raise ValueError("coupling 計測はグループ内 period 一様が前提 (イベント整列が壊れる)")
    lop_steps = build_lop_steps(cfg, total, period_val)
    wdir_ctx = make_wdir_ctx(st, cfg)
    bctx = make_bctx(st, cfg)

    # 実験(5) coupling_ab: タスク境界 (period, 2*period, ...) 直後 postswitch_n ステップの
    # 平均二乗誤差 (§②「新タスク切り替え直後の予測誤差」)。coupling キーが無ければ全て無効。
    coupling_cfg = cfg.get("coupling")
    sw_list = switch_steps(period_val, total) if coupling_cfg and period_val else []
    postswitch_n = (coupling_cfg or {}).get("postswitch_n", 10)
    sw_ptr = 0
    while sw_ptr < len(sw_list) and sw_list[sw_ptr] < start_step:
        sw_ptr += 1
    active_switch = None
    post_acc = torch.zeros(st["R"], device=device)
    post_count = 0
    postswitch_rows = []

    loss_rows, lop_rows = [], []
    loss_acc = torch.zeros(st["R"], device=device)
    t0 = time.time()

    batch, batch_n = st["batch"], st["batch_n"]
    noise_sd = st["noise_sd"]
    snap_set = set(snapshot_steps)
    # probe=None なら空集合 -> 判定は毎ステップ False で既存挙動と完全一致 [posreset_0819 §5]
    probe_set = set(probe_steps) if probe is not None else set()
    # タスク境界の flip 後・当該サンプルでの更新前を読むための opt-in hook
    # [dynrepair_0826 §3.3, §8]。既定は完全な no-op。
    pre_update_probe_set = (
        set(pre_update_probe_steps) if pre_update_probe is not None else set()
    )

    if start_step > 0:
        # resume 直後 (介入後・タスク切替前) の状態を step=start_step として記録。
        # 連続 run の同 step 行と直接比較できる (none アームなら bit 一致 = S1)。
        assert start_step % C["loss_bin"] == 0, "start_step は loss_bin 境界であること"
        x_ev, y_ev = eval_batch(st)
        x_ev_in = x_ev - cmask[None] * st["running_mean"][None]
        m = compute_lop_metrics(net, x_ev_in, y_ev, cfg, wdir_ctx=wdir_ctx, bctx=bctx)
        lop_rows.append((start_step, {k: v.cpu().numpy().copy() for k, v in m.items()}))

    for t in range(start_step, total):
        if t in ckpt_set:
            save_ckpt(st, t, outdir)
        if t in snap_set:
            save_snapshot(st, t, outdir)
        if t in probe_set:
            probe(st, t)

        if sw_ptr < len(sw_list) and t == sw_list[sw_ptr]:
            active_switch = sw_list[sw_ptr]
            sw_ptr += 1
            post_acc = torch.zeros(st["R"], device=device)
            post_count = 0

        if st["exp"] == "B":
            teacher.t = env.t
            teacher.maybe_resample()

        if batch == 1:
            x_raw = env.step()                               # [R,d]
            y = teacher(x_raw)                               # [R]
            if noise_sd > 0:                                 # 学習信号のみ汚す (eval は clean)
                y = y + noise_sd * torch.randn(y.shape, generator=st["gens"]["noise"],
                                               device=device)
            if (t + 1) in pre_update_probe_set:
                pre_update_probe(st, t + 1)

            x_in = x_raw - cmask * st["running_mean"]
            st["running_mean"].mul_(1 - alpha).add_(alpha * x_raw)

            pre, a, yhat = net.forward(x_in)
            delta = yhat - y
            gW, gb, gv, gc = net.grads(x_in, pre, a, delta)
            net.sgd_step(lr, gW, gb, gv, gc)
            if st["method"]["name"] not in ("none", "leaky"):
                apply_method(st, a)

            loss_acc += delta ** 2
        else:
            # ミニバッチ/フルバッチ: バッチ平均勾配で 1 ステップ更新
            if batch == "full" and st["exp"] == "A":
                x_raw = env.full_support()                   # [2^(m-f),R,d] 厳密列挙
            else:
                x_raw = env.step_batch(batch_n)              # [B,R,d]
            y = teacher(x_raw)                               # [B,R]
            if noise_sd > 0:
                y = y + noise_sd * torch.randn(y.shape, generator=st["gens"]["noise"],
                                               device=device)
            if (t + 1) in pre_update_probe_set:
                pre_update_probe(st, t + 1)

            x_in = x_raw - cmask[None] * st["running_mean"][None]
            st["running_mean"].mul_(1 - alpha).add_(alpha * x_raw.mean(dim=0))

            pre, a, yhat = net.forward_batch(x_in)
            delta = yhat - y
            gW, gb, gv, gc = net.grads_batch(x_in, pre, a, delta)
            net.sgd_step(lr, gW.mean(0), gb.mean(0), gv.mean(0), gc.mean(0))

            loss_acc += (delta ** 2).mean(dim=0)

        if active_switch is not None:
            d2 = delta ** 2 if batch == 1 else (delta ** 2).mean(dim=0)
            post_acc += d2
            post_count += 1
            if post_count == postswitch_n:
                postswitch_rows.append((active_switch, post_acc.cpu().numpy().copy() / postswitch_n))
                active_switch = None

        if (t + 1) % C["loss_bin"] == 0:
            loss_rows.append((t + 1, (loss_acc / C["loss_bin"]).cpu().numpy().copy()))
            loss_acc.zero_()

        if (t + 1) in lop_steps or t == 0:
            x_ev, y_ev = eval_batch(st)
            x_ev_in = x_ev - cmask[None] * st["running_mean"][None]
            m = compute_lop_metrics(net, x_ev_in, y_ev, cfg, wdir_ctx=wdir_ctx, bctx=bctx)
            lop_rows.append((t + 1 if t > 0 else 0, {k: v.cpu().numpy().copy() for k, v in m.items()}))

    if total in probe_set:                 # 末尾 step はループ本体を通らないので補う
        probe(st, total)
    if total in snap_set:                  # 同上 (既存呼び出しは t_int < total なので影響なし)
        save_snapshot(st, total, outdir)
    if total in ckpt_set:
        save_ckpt(st, total, outdir)

    elapsed = time.time() - t0
    write_logs(st, loss_rows, lop_rows, outdir, postswitch_rows=postswitch_rows,
               total=total)
    return st, elapsed


def write_logs(st, loss_rows, lop_rows, outdir, postswitch_rows=None, total=None):
    os.makedirs(outdir, exist_ok=True)
    gname = st["gname"]
    ids = [r["run_id"] for r in st["runs"]]

    # CBP の累積 reset 数 (理論値 rho*width*steps との照合用) [cbp_harm_0815 S3]
    if st.get("cbp") is not None:
        exp_n = float(st["method"].get("rho", 0.0)) * st["width"] * (total or 0)
        with open(os.path.join(outdir, f"cbp_stats_{gname}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run_id", "n_reset", "expected", "rho", "width", "steps"])
            nr = st["cbp"]["n_reset"].cpu().numpy()
            for i, rid in enumerate(ids):
                w.writerow([rid, f"{nr[i]:.0f}", f"{exp_n:.2f}",
                            st["method"].get("rho", 0.0), st["width"], total or 0])

    with open(os.path.join(outdir, f"online_loss_{gname}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "run_id", "loss"])
        for step, arr in loss_rows:
            for i, rid in enumerate(ids):
                w.writerow([step, rid, f"{arr[i]:.6g}"])

    if lop_rows:
        keys = list(lop_rows[0][1].keys())
        with open(os.path.join(outdir, f"lop_metrics_{gname}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "run_id"] + keys)
            for step, m in lop_rows:
                for i, rid in enumerate(ids):
                    w.writerow([step, rid] + [f"{m[k][i]:.6g}" for k in keys])

    if postswitch_rows:
        with open(os.path.join(outdir, f"postswitch_err_{gname}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["switch_step", "run_id", "post_err"])
            for switch_step, arr in postswitch_rows:
                for i, rid in enumerate(ids):
                    w.writerow([switch_step, rid, f"{arr[i]:.6g}"])

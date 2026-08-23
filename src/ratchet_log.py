"""ratchet_log: 整流モデルの時間発展ロギング走。

  OMP_NUM_THREADS=1 .venv/bin/python -m src.ratchet_log --config configs/ratchet_log_0819.yaml
  OMP_NUM_THREADS=1 .venv/bin/python -m src.ratchet_log --smoke     # 0->50k, seed 1 本
  OMP_NUM_THREADS=1 .venv/bin/python -m src.ratchet_log --s2        # 無擾乱チェックのみ

train_group の probe フック [posreset_0819 §5] に**読み取り専用**の記録関数を差し、
condA A_w100 の 32 パターン厳密期待値を細密グリッド上に落とす。介入アームは無い。

**probe の無擾乱性がこの実験の生命線** (§7 S2)。特に `envs.SCREnv.full_support()` は
`maybe_flip()` を呼んで env.t を進める「学習ループ用」の関数であり、probe から呼ぶと
タスク境界が前倒しになって軌道そのものが変わる。ここでは `full_support_ro()` を別途
用意し、flip_state と patterns を**読むだけ**にしてある。

記録量は §3.4 の通り。計算は float64、保存は float32。
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import yaml

from .common import (ROOT, load_config, pick_device, build_runs, group_runs,
                     resolve_outdir, switch_steps)
from .train import setup_group, train_group

# 保存するユニット行列 [n_rec, R, h] と run 配列 [n_rec, R, ...] のキー
# F_gate は F_self + F_rest と数学的に同値だが**別に保存する**。δ_self,i = v_i·a_i と
# δ_rest,i = δ − δ_self,i は個別には大きく和が小さい (δ は残差) ため、float32 に丸めてから
# 足すと桁落ちで有効数字が数桁飛ぶ (実測: 相対 7e-5)。合計を使う解析はこの列を読むこと。
LEGACY_UNIT_KEYS = ["cos_u_mu", "p_hat", "w_norm", "b", "v",
                    "F_self", "F_rest", "F_gate"]
UNIT_KEYS = LEGACY_UNIT_KEYS + [
    # mu titration [中心主張 v3 作業5]。M は自由 bit 部の最大変動、s は平均場。
    "M", "s", "b_plus_M", "cos_crit", "delta_b_field", "delta_wmu_field",
]
RUN_VEC_KEYS = {"G": "m", "flip_state": "f"}          # [n_rec, R, dim]
RUN_SCA_KEYS = ["E_delta", "mu_norm", "ratio_mu_cov", "cos_G_mu", "G_dot_mu",
                "eval_loss_exact"]

# `max(pre) = s + M` は float64 の有限和どうしの比較だが、加算順による丸めを許容する。
# mu titration spec S5a--c の固定閾値。
FIELD_IDENTITY_ATOL = 1e-12
S3_IDENTITY_ATOL = 1e-12
S3_EXPECTED_EVAL_N = 2000
S3_EXPECTED_SUPPORT = 32


# ---------------------------------------------------------------- 読み取り専用の厳密列挙

def full_support_ro(env):
    """`envs.SCREnv.full_support()` の**読み取り専用**版 [§3.4 / §7 S2]。

    本家は先頭で `maybe_flip()` を呼び `self.t += 1` する (学習ループが 1 step 進む
    前提の関数)。probe から本家を呼ぶと (i) タスク境界が 1 step 早まり (ii) flip 用の
    乱数が余分に消費され、probe の有無で軌道が変わる。ここは flip_state / patterns を
    読むだけで env の状態に一切書き込まない。"""
    P = env.patterns.shape[0]
    flip = env.flip_state.unsqueeze(0).expand(P, -1, -1)      # [P,R,f]
    rnd = env.patterns[:, None, :].expand(-1, env.R, -1)      # [P,R,m-f]
    return torch.cat([flip, rnd], dim=2)                      # [P,R,m]


def teacher_f64(teacher, X):
    """`envs.LTUTarget.__call__` の float64 版 (式は逐語的に同一)。
    tau は beta*(m+1)-S で非整数になるため、境界判定を float32 の丸めに委ねない。
    out_scale [teachw_0820 §3] も本家と同じく cout 込みの全体に掛ける (既定 1.0)。"""
    pre = torch.einsum("rhm,prm->prh", teacher.W.double(), X) + teacher.b.double()
    h = (pre >= teacher.tau.double()).double()
    y = (h * teacher.v.double()).sum(dim=-1) + teacher.cout.double()
    return y * float(getattr(teacher, "out_scale", 1.0))


def exact_record(st, as_f64=False, _with_sanity=False):
    """1 記録点ぶんの 32 パターン厳密期待値 [§3.4]。全て float64 で計算。

    返り値は numpy の dict (run 配列 [R,*] / unit 行列 [R,h])。既定は保存用の float32
    だが、as_f64=True で計算精度そのままの float64 を返す (Phase 0 の 1e-10 判定用。
    float32 に丸めた値と float64 の参照実装を比べると、丸め自体で 1e-8 台の相対誤差が
    出て判定が意味を失う)。

    記号 (net の勾配は nets.VecMLP.grads と同じ係数 2 込み):
      δ = ŷ − y、G = E[δx_in]、µ = E[x_in]
      F_i^gate = −2η v_i E[δ·gate_i·x_in]、F_i^ungate = −2η v_i G
      δ = δ_self,i + δ_rest,i、δ_self,i = v_i σ(w_iᵀx_in+b_i)
    std では µ = flip_state ‖ 0.5·1、centered では
    µ = E[x] − running_mean。いずれも 32 パターンの x_in から数値計算する。
    F の µ 射影は**単位ベクトル** µ/‖µ‖ に対して取る。‖µ‖ は mu_norm として保存する
    ので生の内積は事後復元できる。"""
    env, net, teacher = st["env"], st["net"], st["teacher"]
    with torch.no_grad():
        X = full_support_ro(env).double()                     # [P,R,m]
        y = teacher_f64(teacher, X)                           # [P,R]

        # train_group の同じ step の forward と同じく、更新前の running_mean を使う。
        # µ は enc に依らず、以下の x_in の 32 パターン平均から数値計算する。
        cmask = st["centered"][:, None].double()              # [R,1]
        x_in = X - cmask[None] * st["running_mean"].double()[None]

        W, b = net.W.double(), net.b.double()
        v, c = net.v.double(), net.c.double()
        lr = st["lr"].double()                                # [R]

        pre = torch.einsum("rhd,prd->prh", W, x_in) + b       # [P,R,h]
        gate = (pre > 0).double()
        a = torch.relu(pre)
        yhat = (a * v).sum(dim=-1) + c                        # [P,R]
        delta = yhat - y                                      # [P,R]

        # --- run レベル
        mu = x_in.mean(dim=0)                                 # [R,m] = E[x_in]
        mu_norm = mu.norm(dim=1)                              # [R]
        mu_u = mu / mu_norm.clamp_min(1e-300)[:, None]
        G = (delta[:, :, None] * x_in).mean(dim=0)            # [R,m]
        E_delta = delta.mean(dim=0)                           # [R]
        cov_dx = G - E_delta[:, None] * mu                    # Cov(δ,x) = E[δx]−E[δ]E[x]
        ratio = (E_delta.abs() * mu_norm) / cov_dx.norm(dim=1).clamp_min(1e-300)
        G_dot_mu = (G * mu_u).sum(dim=1)                      # [R] 単位 µ̂ への射影
        cos_G_mu = G_dot_mu / G.norm(dim=1).clamp_min(1e-300)

        # --- unit レベル
        w_norm = W.norm(dim=2)                                # [R,h]
        cos_u_mu = torch.einsum("rhd,rd->rh", W, mu_u) / w_norm.clamp_min(1e-300)
        p_hat = gate.mean(dim=0)                              # [R,h] 厳密ゲート率

        xdm = (x_in * mu_u[None]).sum(dim=-1)                 # [P,R] x·µ̂/‖µ̂‖
        d_self = v[None] * a                                  # [P,R,h] δ_self,i
        d_rest = delta[:, :, None] - d_self                   # [P,R,h]
        pref = -2.0 * lr[:, None] * v                         # [R,h] = −2η v_i
        F_self = pref * (d_self * gate * xdm[:, :, None]).mean(dim=0)
        F_rest = pref * (d_rest * gate * xdm[:, :, None]).mean(dim=0)
        # 合計は float64 のまま作る (float32 の F_self + F_rest では桁落ちする)
        F_gate = pref * (delta[:, :, None] * gate * xdm[:, :, None]).mean(dim=0)

        # --- mu titration の unit-level 十分統計 [中心主張 v3 作業5]
        # x_in = mu + delta_x。flip 部は固定なので delta_x=0、自由 bit 部は ±1/2。
        # 従って max_delta w.delta = (1/2)||w_free||_1 を全32支持点を列挙せずにも書ける。
        free = slice(int(env.f), int(env.m))
        M = 0.5 * W[:, :, free].abs().sum(dim=2)              # [R,h]
        s = torch.einsum("rhd,rd->rh", W, mu) + b            # [R,h] = w.mu+b
        b_plus_M = b + M
        cos_crit = -(b_plus_M) / (
            w_norm * mu_norm[:, None]).clamp_min(1e-300)

        # 期待 full-support SGD update の bias と mu 方向成分。
        # delta_wmu_field は Δw.mu（単位 mu 方向への射影ではない）。
        delta_b_field = pref * (delta[:, :, None] * gate).mean(dim=0)
        xdmu = (x_in * mu[None]).sum(dim=-1)                  # [P,R] = x_in.mu
        delta_wmu_field = pref * (
            delta[:, :, None] * gate * xdmu[:, :, None]).mean(dim=0)

        # S5c の独立な縮約順: full-support の ΔW, Δb を先に作り、最後に Δs=Δb+ΔW.mu。
        Pn = int(x_in.shape[0])
        delta_W_direct = pref[:, :, None] * torch.einsum(
            "pr,prh,prd->rhd", delta, gate, x_in) / Pn
        delta_b_direct = pref * torch.einsum("pr,prh->rh", delta, gate) / Pn
        delta_s_direct = delta_b_direct + torch.einsum("rhd,rd->rh", delta_W_direct, mu)
        delta_s_fields = delta_b_field + delta_wmu_field

        # probe sanity: 解析式 M と32支持点の max、および gate=0 の恒等式を同一状態で検査。
        max_pre = pre.max(dim=0).values
        max_delta_support = torch.einsum(
            "rhd,prd->prh", W, x_in - mu[None]).max(dim=0).values
        field_max = s + M
        p_zero = gate.sum(dim=0) == 0
        pred_zero = field_max <= 0
        # |field_max|<=atol の点は丸め順だけでは符号を確定できない。直接一致数も別に残す。
        mismatch_tol = ((p_zero & (field_max > FIELD_IDENTITY_ATOL)) |
                        ((~p_zero) & (field_max <= -FIELD_IDENTITY_ATOL)))
        denom_raw = w_norm * mu_norm[:, None]
        finite_denom = torch.isfinite(denom_raw) & (denom_raw > 0)
        if bool(finite_denom.any()):
            cos_ref = -(b_plus_M[finite_denom] / denom_raw[finite_denom])
            cos_formula_err = float((cos_crit[finite_denom] - cos_ref).abs().max().item())
        else:
            cos_formula_err = 0.0
        tracked_tensors = (G, E_delta, mu_norm, ratio, cos_G_mu, G_dot_mu, delta,
                           cos_u_mu, p_hat, w_norm, b, v, F_self, F_rest, F_gate,
                           M, s, b_plus_M, cos_crit, delta_b_field, delta_wmu_field)
        sanity = dict(
            n_units=int(p_zero.numel()),
            n_zero=int(p_zero.sum().item()),
            n_mismatch_raw=int((p_zero != pred_zero).sum().item()),
            n_mismatch_beyond_tol=int(mismatch_tol.sum().item()),
            n_near_boundary=int((field_max.abs() <= FIELD_IDENTITY_ATOL).sum().item()),
            max_pre_identity_abs_err=float((max_pre - field_max).abs().max().item()),
            max_M_support_abs_err=float((max_delta_support - M).abs().max().item()),
            max_cos_crit_formula_abs_err=cos_formula_err,
            n_finite_cos_crit_denominator=int(finite_denom.sum().item()),
            max_delta_s_field_abs_err=float((delta_s_fields - delta_s_direct).abs().max().item()),
            max_p_hat_quantization_abs_err=float(
                (p_hat * Pn - torch.round(p_hat * Pn)).abs().max().item()),
            n_nonfinite_all_stats=int(sum((~torch.isfinite(t)).sum().item()
                                          for t in tracked_tensors)),
        )

        dt = np.float64 if as_f64 else np.float32
        cv = lambda t: t.detach().cpu().numpy().astype(dt)
        out = dict(
            G=cv(G), flip_state=cv(env.flip_state.double()),
            E_delta=cv(E_delta), mu_norm=cv(mu_norm), ratio_mu_cov=cv(ratio),
            cos_G_mu=cv(cos_G_mu), G_dot_mu=cv(G_dot_mu),
            eval_loss_exact=cv((delta ** 2).mean(dim=0)),
            cos_u_mu=cv(cos_u_mu), p_hat=cv(p_hat), w_norm=cv(w_norm),
            b=cv(b), v=cv(v), F_self=cv(F_self), F_rest=cv(F_rest), F_gate=cv(F_gate),
            M=cv(M), s=cv(s), b_plus_M=cv(b_plus_M), cos_crit=cv(cos_crit),
            delta_b_field=cv(delta_b_field), delta_wmu_field=cv(delta_wmu_field))
        return (out, sanity) if _with_sanity else out


# ---------------------------------------------------------------- 記録グリッド

def record_steps(total, period, half_window, bulk_every):
    """§3.3 のグリッド: 境界 ±half_window を毎 step + それ以外を bulk_every ごと。

    境界は switch_steps と同じ {period, 2·period, ...}。ただし **t=total の境界では
    flip が起きない** (train_group のループは range(start, total) で、t=total の
    反復自体が無い) ので、実現する遷移は 1 つ少ない。ここは記録点の集合を返すだけで、
    遷移の勘定は解析側 (figures_ratchet_log) が行う。"""
    steps = set(range(0, total + 1, bulk_every))
    for s0 in switch_steps(period, total):
        steps.update(range(max(0, s0 - half_window), min(total, s0 + half_window) + 1))
    steps.add(0)
    steps.add(total)
    return sorted(steps)


def record_key_selection(ratchet_cfg):
    """保存列を config から選ぶ（未指定なら従来 schema の列だけ）。

    長い mu titration では解析に不要な旧 F 分解などを落として NPZ / 常駐メモリを節約
    できる。exact_record 自体は常に全統計を計算するため、sanity の内容は列選択に依存しない。
    `run_vec_keys` は初期案との互換 alias、正本は `run_vector_keys`。"""
    def take(name, allowed, alias=None, default=None):
        raw = ratchet_cfg.get(name)
        if raw is None and alias:
            raw = ratchet_cfg.get(alias)
        vals = list(allowed if default is None else default) if raw is None else list(raw)
        if len(vals) != len(set(vals)):
            raise ValueError(f"ratchet.{name} に重複キー: {vals}")
        bad = sorted(set(vals) - set(allowed))
        if bad:
            raise ValueError(f"ratchet.{name} の未知キー: {bad}; 選択肢={list(allowed)}")
        return vals

    return dict(
        # 旧 config の NPZ schema / 常駐メモリは不変。新列は config で明示した走だけ保存。
        unit_keys=take("unit_keys", UNIT_KEYS, default=LEGACY_UNIT_KEYS),
        run_vector_keys=take("run_vector_keys", RUN_VEC_KEYS, alias="run_vec_keys"),
        run_scalar_keys=take("run_scalar_keys", RUN_SCA_KEYS),
    )


class Recorder:
    """probe(st, step) として train_group に渡す読み取り専用アキュムレータ。

    記録点は事前に確定しているので配列を先に確保する (np.stack の一時 2 倍を避ける)。
    s3_steps に指定した記録点では S3 用に32 supportを独立再列挙し、固定evalの
    pattern頻度で再重み付けした決定論的照合結果も控える (addendum2)。この計算は
    generator を消費せず env.t も進めない。"""

    def __init__(self, steps, R, h, m, f, s3_steps=(), unit_keys=None,
                 run_vector_keys=None, run_scalar_keys=None):
        self.steps = np.asarray(steps, dtype=np.int64)
        self.index = {int(s): i for i, s in enumerate(self.steps)}
        n = len(self.steps)
        self.unit_keys = list(LEGACY_UNIT_KEYS if unit_keys is None else unit_keys)
        self.run_vector_keys = list(RUN_VEC_KEYS if run_vector_keys is None
                                    else run_vector_keys)
        self.run_scalar_keys = list(RUN_SCA_KEYS if run_scalar_keys is None
                                    else run_scalar_keys)
        self.buf = {k: np.zeros((n, R, h), dtype=np.float32) for k in self.unit_keys}
        dims = {"m": m, "f": f}
        for k in self.run_vector_keys:
            self.buf[k] = np.zeros((n, R, dims[RUN_VEC_KEYS[k]]), dtype=np.float32)
        for k in self.run_scalar_keys:
            self.buf[k] = np.zeros((n, R), dtype=np.float32)
        # S4 は保存列から flip_state を外しても必ず実施する。保存する場合は同じ配列を共有。
        self.flip_state_sanity = self.buf.get(
            "flip_state", np.zeros((n, R, f), dtype=np.float32))
        # S5: 記録点ごとの全 R*h unit を集約するための小さな診断配列。
        self.field_n_units = np.zeros(n, dtype=np.int64)
        self.field_n_zero = np.zeros(n, dtype=np.int64)
        self.field_mismatch_raw = np.zeros(n, dtype=np.int64)
        self.field_mismatch_tol = np.zeros(n, dtype=np.int64)
        self.field_near_boundary = np.zeros(n, dtype=np.int64)
        self.field_pre_err = np.zeros(n, dtype=np.float64)
        self.field_M_err = np.zeros(n, dtype=np.float64)
        self.field_cos_crit_err = np.zeros(n, dtype=np.float64)
        self.field_n_finite_cos_denom = np.zeros(n, dtype=np.int64)
        self.field_delta_s_err = np.zeros(n, dtype=np.float64)
        self.p_hat_quant_err = np.zeros(n, dtype=np.float64)
        self.n_nonfinite_stats = np.zeros(n, dtype=np.int64)
        self.filled = np.zeros(n, dtype=bool)
        self.n_calls = 0
        self.step0_repro_hash = None
        self.s3_steps = set(int(s) for s in s3_steps)
        self.s3 = {}                        # step -> deterministic support/eval record

    def __call__(self, st, step):
        i = self.index.get(int(step))
        if i is None:                       # probe_steps と一致するはずだが保険
            return
        if int(step) == 0 and self.step0_repro_hash is None:
            self.step0_repro_hash = reproducibility_hash(st)
        rec, fs = exact_record(st, _with_sanity=True)
        for k, arr in self.buf.items():
            arr[i] = rec[k]
        if "flip_state" not in self.buf:
            self.flip_state_sanity[i] = rec["flip_state"]
        self.field_n_units[i] = fs["n_units"]
        self.field_n_zero[i] = fs["n_zero"]
        self.field_mismatch_raw[i] = fs["n_mismatch_raw"]
        self.field_mismatch_tol[i] = fs["n_mismatch_beyond_tol"]
        self.field_near_boundary[i] = fs["n_near_boundary"]
        self.field_pre_err[i] = fs["max_pre_identity_abs_err"]
        self.field_M_err[i] = fs["max_M_support_abs_err"]
        self.field_cos_crit_err[i] = fs["max_cos_crit_formula_abs_err"]
        self.field_n_finite_cos_denom[i] = fs["n_finite_cos_crit_denominator"]
        self.field_delta_s_err[i] = fs["max_delta_s_field_abs_err"]
        self.p_hat_quant_err[i] = fs["max_p_hat_quantization_abs_err"]
        self.n_nonfinite_stats[i] = fs["n_nonfinite_all_stats"]
        self.filled[i] = True
        self.n_calls += 1
        if int(step) in self.s3_steps:
            self.s3[int(step)] = _s3_support_record(st, rec["p_hat"])

    def check_complete(self):
        miss = np.flatnonzero(~self.filled)
        if len(miss):
            raise RuntimeError(f"未記録の grid 点が {len(miss)} 個: "
                               f"{self.steps[miss][:10]} ...")


def write_logs(rec, runs, outdir, center_alpha=None):
    """seed ごとに logs/seed{k}.npz を書く [§3.4]。"""
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    paths = []
    for i, r in enumerate(runs):
        out = dict(step=rec.steps, run_id=np.array(r["run_id"]), seed=np.int64(r["seed"]),
                   lr=np.float32(r["lr"]), period=np.int64(r["period"]),
                   width=np.int64(r["width"]))
        if center_alpha is not None:
            out["center_alpha"] = np.float32(center_alpha)
        for k, arr in rec.buf.items():
            out[k] = arr[:, i]
        p = os.path.join(logdir, f"seed{r['seed']}.npz")
        np.savez_compressed(p, **out)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- サニティ (§7)

def _sha(t):
    return hashlib.sha256(
        np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def state_hash(st):
    """net と env の全テンソルの sha256 (S2 の bit 一致判定用)。"""
    hs = {f"net.{k}": _sha(v) for k, v in st["net"].state_dict().items()}
    hs["env.flip_state"] = _sha(st["env"].flip_state)
    hs["env.t"] = str(st["env"].t)
    hs["running_mean"] = _sha(st["running_mean"])
    return hs


def reproducibility_hash(st):
    """S6 用: step 0 の net / teacher / env と全 generator 状態の hash。"""
    hs = state_hash(st)
    for k, v in st["teacher"].state_dict().items():
        hs[f"teacher.{k}"] = _sha(v)
    hs["teacher.out_scale"] = str(float(getattr(st["teacher"], "out_scale", 1.0)))
    if hasattr(st["env"], "patterns"):
        hs["env.patterns"] = _sha(st["env"].patterns)
    for name, gen in sorted(st.get("gens", {}).items()):
        hs[f"generator.{name}"] = _sha(gen.get_state())
    return hs


def _binom_tail_ge(n, k, p):
    """P(X >= k), X ~ Binomial(n, p)。PMF漸化式で巨大な ``comb`` を避ける。"""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    term = (1.0 - p) ** n             # P(X=0)
    lower = 0.0
    odds = p / (1.0 - p)
    for j in range(k):
        lower += term
        term *= (n - j) / (j + 1) * odds
    return max(0.0, 1.0 - min(1.0, lower))


def _empirical_p_hat(st):
    """固定 eval バッチ上の経験ゲート率 p̂ [R,h] と バッチサイズ N。
    lop_metrics の open_frac と同一定義 ((pre > 0) の標本平均)。"""
    from .train import eval_batch
    with torch.no_grad():
        x_ev, y_ev = eval_batch(st)
        cmask = st["centered"][:, None].float()
        x_in = x_ev - cmask[None] * st["running_mean"][None]
        pre, _, _ = st["net"].forward_batch(x_in)
        return (pre > 0).float().mean(dim=0).cpu().numpy(), int(x_ev.shape[0])


def _s3_support_record(st, p_exact):
    """addendum2 S3: support再列挙とfixed eval再重み付けをfloat64で独立計算。

    `exact_record` の ``full_support_ro`` / preactivation は再利用せず、5-bit supportを
    ここで作り直す。direct evalも ``net.forward_batch`` を通さず同じfloat64パラメータの
    手書きeinsumで評価し、実現pattern頻度によるsupport再重み付けと照合する。"""
    if st.get("exp") != "A":
        raise ValueError("S3 support再重み付けは condA 専用")
    env, net = st["env"], st["net"]
    q = int(env.m - env.f)
    P = 2 ** q
    with torch.no_grad():
        device = st["device"]
        code = torch.arange(P, device=device, dtype=torch.long)[:, None]
        bit = torch.arange(q, device=device, dtype=torch.long)[None, :]
        support_rnd = ((code >> bit) & 1).double()                 # [P,q]

        flip_support = env.flip_state.double()[None].expand(P, -1, -1)
        rnd_support = support_rnd[:, None, :].expand(-1, st["R"], -1)
        x_support = torch.cat([flip_support, rnd_support], dim=2)  # [P,R,m]
        cmask = st["centered"][:, None].double()
        running_mean = st["running_mean"].double()
        x_support_in = x_support - cmask[None] * running_mean[None]

        W, b = net.W.double(), net.b.double()
        pre_support = torch.einsum("rhd,prd->prh", W, x_support_in) + b
        gate_support = (pre_support > 0).double()                  # [P,R,h]
        p_uniform = gate_support.mean(dim=0)                       # [R,h]

        eval_rnd = st["eval_fixed"].double()                      # [N,q]
        N = int(eval_rnd.shape[0])
        support_match = (eval_rnd[:, None, :] == support_rnd[None]).all(dim=2)
        row_match_count = support_match.sum(dim=1)                 # [N]
        pattern_counts = support_match.sum(dim=0)                  # [P]

        flip_eval = env.flip_state.double()[None].expand(N, -1, -1)
        rnd_eval = eval_rnd[:, None, :].expand(-1, st["R"], -1)
        x_eval = torch.cat([flip_eval, rnd_eval], dim=2)
        x_eval_in = x_eval - cmask[None] * running_mean[None]
        pre_eval = torch.einsum("rhd,nrd->nrh", W, x_eval_in) + b
        gate_eval = (pre_eval > 0).double()
        p_empirical = gate_eval.mean(dim=0)                        # [R,h]
        weights = pattern_counts.double() / max(N, 1)
        p_reweighted = (gate_support * weights[:, None, None]).sum(dim=0)

        p_exact_np = np.asarray(p_exact, dtype=np.float64)
        p_uniform_np = p_uniform.cpu().numpy()
        shape_ok = bool(p_exact_np.shape == p_uniform_np.shape ==
                        tuple(p_empirical.shape) == tuple(p_reweighted.shape))
        exact_uniform_err = (float(np.abs(p_exact_np - p_uniform_np).max())
                             if shape_ok else float("inf"))
        empirical_reweighted_err = float(
            (p_empirical - p_reweighted).abs().max().item())
        degenerate = (p_uniform == 0) | (p_uniform == 1)
        degenerate_err = (float((p_empirical[degenerate] - p_uniform[degenerate])
                                .abs().max().item())
                          if bool(degenerate.any()) else 0.0)

        return dict(
            p_exact=p_exact_np,
            p_uniform=p_uniform_np,
            p_empirical=p_empirical.cpu().numpy(),
            p_reweighted=p_reweighted.cpu().numpy(),
            gate_support=gate_support.cpu().numpy().astype(np.uint8),
            support_match=support_match.cpu().numpy().astype(np.uint8),
            exact_uniform_max_abs_err=exact_uniform_err,
            empirical_reweighted_max_abs_err=empirical_reweighted_err,
            shape_ok=shape_ok,
            N=N, q=q, P=P,
            pattern_counts=pattern_counts.cpu().numpy().astype(np.int64),
            pattern_count_sum=int(pattern_counts.sum().item()),
            n_bad_row_support_matches=int((row_match_count != 1).sum().item()),
            row_support_match_min=int(row_match_count.min().item()) if N else 0,
            row_support_match_max=int(row_match_count.max().item()) if N else 0,
            n_degenerate=int(degenerate.sum().item()),
            max_degenerate_err=degenerate_err,
        )


def check_s3(rec):
    """S3: 3記録点でexact supportと固定evalを決定論的に照合する (addendum2)。

    各点で32 supportをfloat64で独立再列挙した ``p_uniform`` をlogger本体の
    ``p_exact`` と照合する。さらに固定eval 2000行の5-bit pattern実現数でsupport gateを
    再重み付けし、直接float64 forwardした ``p_empirical`` と照合する。各eval行がsupportの
    ちょうど1点に対応すること、count総和、退化unitの厳密一致も必須である。

    従来の z / median / exceedance / binomial tail は互換診断として記録するだけで、
    相関したunit familyに閾値を置かないためPASS/FAILには一切使用しない。"""
    per, zs = [], []
    max_exact_err = 0.0
    max_reweight_err = 0.0
    max_uniform_reconstruction_err = 0.0
    max_reweight_reconstruction_err = 0.0
    max_degen_err = 0.0
    n_deg = 0
    n_bad_rows = 0
    all_shapes = True
    all_finite = True
    all_discrete = True
    all_dimensions = True
    all_unique_rows = True
    all_count_sums = True
    all_counts_match_rows = True
    eval_sizes, support_sizes, free_bits = [], [], []

    for step in sorted(rec.s3):
        item = rec.s3[step]
        p_ex = np.asarray(item["p_exact"], dtype=np.float64)
        p_uniform = np.asarray(item["p_uniform"], dtype=np.float64)
        p_emp = np.asarray(item["p_empirical"], dtype=np.float64)
        p_reweighted = np.asarray(item["p_reweighted"], dtype=np.float64)
        N, q, P = int(item["N"]), int(item["q"]), int(item["P"])
        counts_raw = np.asarray(item["pattern_counts"])
        gate_support_raw = np.asarray(item["gate_support"])
        support_match_raw = np.asarray(item["support_match"])
        counts = counts_raw.astype(np.int64, copy=False)
        gate_support = gate_support_raw.astype(np.float64, copy=False)
        support_match = support_match_raw.astype(np.int64, copy=False)

        p_shape_ok = bool(p_ex.ndim == 2 and
                          p_ex.shape == p_uniform.shape == p_emp.shape ==
                          p_reweighted.shape)
        gate_shape_ok = bool(p_ex.ndim == 2 and
                             gate_support.shape == (P, *p_ex.shape))
        match_shape_ok = bool(support_match.shape == (N, P))
        shape_ok = bool(item.get("shape_ok", False) and p_shape_ok and
                        gate_shape_ok and match_shape_ok and counts.shape == (P,))
        finite_ok = bool(p_shape_ok and all(np.isfinite(x).all()
                                           for x in (p_ex, p_uniform, p_emp,
                                                     p_reweighted, gate_support)))
        integral = lambda a: (np.issubdtype(a.dtype, np.integer) or
                              np.issubdtype(a.dtype, np.bool_))
        discrete_ok = bool(
            integral(counts_raw) and integral(gate_support_raw) and
            integral(support_match_raw) and (counts >= 0).all() and
            ((gate_support == 0) | (gate_support == 1)).all() and
            ((support_match == 0) | (support_match == 1)).all()
        )
        derive_ok = bool(p_shape_ok and gate_shape_ok and counts.shape == (P,) and
                         finite_ok and discrete_ok and N > 0)
        if derive_ok:
            p_uniform_from_gates = gate_support.mean(axis=0)
            p_reweighted_from_gates = (
                gate_support * (counts.astype(np.float64) / N)[:, None, None]
            ).sum(axis=0)
            exact_err = float(np.abs(p_ex - p_uniform_from_gates).max())
            reweight_err = float(np.abs(p_emp - p_reweighted_from_gates).max())
            uniform_reconstruction_err = float(
                np.abs(p_uniform - p_uniform_from_gates).max())
            reweight_reconstruction_err = float(
                np.abs(p_reweighted - p_reweighted_from_gates).max())
            degen = ((p_uniform_from_gates == 0.0) |
                     (p_uniform_from_gates == 1.0))
            degen_err = (float(np.abs(
                p_emp[degen] - p_uniform_from_gates[degen]).max())
                if degen.any() else 0.0)
        else:
            exact_err = reweight_err = float("inf")
            uniform_reconstruction_err = reweight_reconstruction_err = float("inf")
            degen = np.zeros_like(p_ex, dtype=bool)
            degen_err = float("inf")

        match_valid = bool(match_shape_ok and discrete_ok)
        if match_valid:
            row_match_counts = support_match.sum(axis=1)
            counts_from_rows = support_match.sum(axis=0)
            bad_rows = int((row_match_counts != 1).sum())
            row_min = int(row_match_counts.min()) if N else 0
            row_max = int(row_match_counts.max()) if N else 0
        else:
            counts_from_rows = np.zeros(P, dtype=np.int64)
            bad_rows, row_min, row_max = N, 0, 0
        recorded_bad_rows = int(item["n_bad_row_support_matches"])
        recorded_row_min = int(item["row_support_match_min"])
        recorded_row_max = int(item["row_support_match_max"])
        counts_match_rows = bool(match_valid and counts.shape == (P,) and
                                 np.array_equal(counts, counts_from_rows))
        count_sum = int(counts.sum())
        recorded_count_sum = int(item["pattern_count_sum"])
        dimensions_ok = bool(N == S3_EXPECTED_EVAL_N and q == 5 and
                             P == S3_EXPECTED_SUPPORT and counts.shape == (P,))
        unique_rows_ok = bool(
            match_valid and bad_rows == recorded_bad_rows == 0 and
            row_min == recorded_row_min == 1 and row_max == recorded_row_max == 1
        )
        count_sum_ok = bool(discrete_ok and count_sum == recorded_count_sum == N)

        # 過去との比較用z診断。ここでのpはlogger exact、経験率は直接eval forward。
        sd = np.sqrt(np.maximum(p_ex * (1.0 - p_ex), 0.0) / max(N, 1))
        if finite_ok:
            z_degen = sd == 0
            z = np.where(z_degen, 0.0,
                         (p_emp - p_ex) / np.maximum(sd, 1e-12))[~z_degen]
        else:
            z = np.zeros(0, dtype=np.float64)
        zs.append(z)

        max_exact_err = max(max_exact_err, exact_err)
        max_reweight_err = max(max_reweight_err, reweight_err)
        max_uniform_reconstruction_err = max(
            max_uniform_reconstruction_err, uniform_reconstruction_err)
        max_reweight_reconstruction_err = max(
            max_reweight_reconstruction_err, reweight_reconstruction_err)
        max_degen_err = max(max_degen_err, degen_err)
        n_deg += int(degen.sum()) if shape_ok else 0
        n_bad_rows += bad_rows
        all_shapes = all_shapes and shape_ok
        all_finite = all_finite and finite_ok
        all_discrete = all_discrete and discrete_ok
        all_dimensions = all_dimensions and dimensions_ok
        all_unique_rows = all_unique_rows and unique_rows_ok
        all_count_sums = all_count_sums and count_sum_ok
        all_counts_match_rows = all_counts_match_rows and counts_match_rows
        eval_sizes.append(N)
        support_sizes.append(P)
        free_bits.append(q)
        per.append(dict(
            step=int(step), N=N, q=q, P=P, shape_ok=shape_ok, finite_ok=finite_ok,
            discrete_arrays_ok=discrete_ok,
            exact_uniform_max_abs_err=exact_err,
            empirical_reweighted_max_abs_err=reweight_err,
            p_uniform_reconstruction_max_abs_err=uniform_reconstruction_err,
            p_reweighted_reconstruction_max_abs_err=reweight_reconstruction_err,
            pattern_counts=counts.tolist(), pattern_count_sum=count_sum,
            recorded_pattern_count_sum=recorded_count_sum,
            pattern_counts_match_support_rows=counts_match_rows,
            n_bad_row_support_matches=bad_rows,
            recorded_n_bad_row_support_matches=recorded_bad_rows,
            row_support_match_min=row_min, row_support_match_max=row_max,
            recorded_row_support_match_min=recorded_row_min,
            recorded_row_support_match_max=recorded_row_max,
            n_z=int(z.size),
            median_abs_z=(round(float(np.median(np.abs(z))), 4) if z.size else None),
            max_abs_z=(round(float(np.abs(z).max()), 4) if z.size else None),
            frac_gt3=(round(float((np.abs(z) > 3).mean()), 5) if z.size else None),
            n_degenerate=int(degen.sum()) if shape_ok else 0,
            max_degenerate_err=degen_err,
        ))

    z = np.concatenate(zs) if zs else np.zeros(0, dtype=np.float64)
    med = float(np.median(np.abs(z))) if z.size else 0.0
    n, k = int(z.size), int((np.abs(z) > 3).sum())
    frac_gt3 = float(k / n) if n else 0.0
    tail = _binom_tail_ge(n, k, 0.0026998)
    n_points = len(rec.s3)
    ok = bool(
        n_points == 3 and all_shapes and all_finite and all_discrete and
        all_dimensions and all_unique_rows and all_count_sums and all_counts_match_rows and
        max_exact_err <= S3_IDENTITY_ATOL and max_reweight_err <= S3_IDENTITY_ATOL and
        max_uniform_reconstruction_err <= S3_IDENTITY_ATOL and
        max_reweight_reconstruction_err <= S3_IDENTITY_ATOL and max_degen_err == 0.0
    )
    return dict(
        s3_pass=ok,
        s3_n_points=n_points,
        s3_identity_atol=S3_IDENTITY_ATOL,
        s3_max_exact_uniform_abs_err=max_exact_err,
        s3_max_empirical_reweighted_abs_err=max_reweight_err,
        s3_max_p_uniform_reconstruction_abs_err=max_uniform_reconstruction_err,
        s3_max_p_reweighted_reconstruction_abs_err=max_reweight_reconstruction_err,
        s3_shape_pass=bool(all_shapes),
        s3_all_values_finite=bool(all_finite),
        s3_discrete_arrays_pass=bool(all_discrete),
        s3_dimensions_pass=bool(all_dimensions),
        s3_eval_rows_all_match_one_support=bool(all_unique_rows),
        s3_n_bad_eval_row_support_matches=int(n_bad_rows),
        s3_pattern_count_sum_all_match_N=bool(all_count_sums),
        s3_pattern_counts_match_support_rows=bool(all_counts_match_rows),
        s3_eval_N=(eval_sizes[0] if len(set(eval_sizes)) == 1 and eval_sizes else None),
        s3_support_P=(support_sizes[0]
                      if len(set(support_sizes)) == 1 and support_sizes else None),
        s3_free_bits_q=(free_bits[0] if len(set(free_bits)) == 1 and free_bits else None),
        s3_n_degenerate=n_deg,
        s3_max_degenerate_err=max_degen_err,
        s3_median_abs_z=round(med, 4),
        s3_n_gt3=k,
        s3_expected_gt3=round(0.0026998 * n, 3),
        s3_frac_gt3=frac_gt3,
        s3_binom_tail_p=round(tail, 5),
        s3_max_abs_z=round(float(np.abs(z).max()), 4) if z.size else None,
        s3_n_z=n,
        s3_criterion=("3 points; q=5, P=32, N=2000; each eval row matches exactly one "
                      "support; sum counts=N; max exact-vs-uniform and empirical-vs-"
                      "reweighted error<=1e-12; degenerate error=0 (addendum2)"),
        s3_note=("median|max|frac(|z|>3) と独立二項tailは互換診断のみ。"
                 "いずれもPASS/FAILには使用しない。"),
        s3_per_step=per,
    )


def check_s4(rec, period):
    """S4: flip_state が変化する step が t ≡ 0 (mod period) の直後だけであること [§7]。

    probe は train_group のループ本体先頭 (env.step() の前) で呼ばれるので、
    境界 B の flip は「記録点 B」と「記録点 B+1」の間で起きる。したがって
    flip_state が動く記録点ペアの左端は必ず period の倍数になる。"""
    fs = rec.flip_state_sanity                                     # [n,R,f]
    changed = (np.abs(np.diff(fs, axis=0)) > 0).any(axis=(1, 2))   # [n-1]
    left = rec.steps[:-1][changed]
    right = rec.steps[1:][changed]
    bad = [int(s) for s in left if s % period != 0]
    # 反転は 1 ビットのみ (maybe_flip の仕様)
    nbits = np.abs(np.diff(fs, axis=0)).sum(axis=2)                # [n-1,R]
    bad_bits = int(((nbits != 0) & (nbits != 1)).sum())
    # 隣接記録点でないペア (バルク粒度をまたぐ) は「1 ビット」を保証できないので除く
    adjacent = (right - left) == 1
    return dict(s4_pass=bool(not bad and bad_bits == 0),
                s4_n_flip_transitions=int(changed.sum()),
                s4_n_adjacent=int(adjacent.sum()),
                s4_bad_steps=bad[:10], s4_n_bad_bitcount=bad_bits)


def check_s5(rec, atol=FIELD_IDENTITY_ATOL):
    """S5: 全記録点で `p_hat==0 iff s+M<=0` と解析式 M を検査する。

    `raw` は浮動小数の符号をそのまま比較した不一致数。PASS はさらに、境界から atol
    より離れた不一致が 0 で、32支持点から得た max(pre) / max(w.delta) と解析式の最大
    絶対誤差が atol 以下であることを要求する。これにより丸めだけで符号が変わり得る
    |s+M|<=atol の点を誤って理論反例とは数えない一方、その個数は隠さず記録する。"""
    n = int(rec.field_n_units.sum())
    raw = int(rec.field_mismatch_raw.sum())
    beyond = int(rec.field_mismatch_tol.sum())
    near = int(rec.field_near_boundary.sum())
    max_pre = float(rec.field_pre_err.max(initial=0.0))
    max_M = float(rec.field_M_err.max(initial=0.0))
    max_cos = float(rec.field_cos_crit_err.max(initial=0.0))
    max_delta_s = float(rec.field_delta_s_err.max(initial=0.0))
    max_p_quant = float(rec.p_hat_quant_err.max(initial=0.0))
    n_nonfinite = int(rec.n_nonfinite_stats.sum())
    n_finite_denom = int(rec.field_n_finite_cos_denom.sum())
    # spec S5b は論理不一致を厳密に 0 と固定。tolerance-aware 値も診断として残すが、
    # raw mismatch を隠して PASS にはしない。
    ok = bool(n > 0 and raw == 0 and beyond == 0 and n_finite_denom > 0 and
              max_pre <= atol and max_M <= atol and max_cos <= atol and
              max_delta_s <= atol and max_p_quant <= atol and n_nonfinite == 0)
    return dict(
        s5_pass=ok,
        s5_n_unit_points=n,
        s5_n_zero=int(rec.field_n_zero.sum()),
        s5_n_mismatch_raw=raw,
        s5_all_points_raw_match=bool(raw == 0),
        s5_n_mismatch_beyond_tol=beyond,
        s5_all_points_match_with_tol=bool(beyond == 0),
        s5_n_near_boundary=near,
        s5_max_pre_identity_abs_err=max_pre,
        s5_max_M_support_abs_err=max_M,
        s5_max_cos_crit_formula_abs_err=max_cos,
        s5_n_finite_cos_crit_denominator=n_finite_denom,
        s5_max_delta_s_field_abs_err=max_delta_s,
        s5_max_p_hat_times_32_integer_abs_err=max_p_quant,
        s5_n_nonfinite_all_stats=n_nonfinite,
        s5_atol=float(atol),
        s5_criterion=("raw論理不一致=0 かつ max|pre_max-(s+M)|<=atol かつ "
                      "max|max_support(w.delta)-M|<=atol かつ有限分母上の "
                      "cos_crit式誤差<=atol かつ直接full-support Δsとの誤差<=atol かつ "
                      "p_hat*32量子化誤差<=atol かつ非有限値0"),
    )


def check_s7(rec, R, expected_steps):
    """S7: shape / finite / p_hat の 1/32 量子化。"""
    actual_steps = int(len(rec.steps))
    max_quant = float(rec.p_hat_quant_err.max(initial=0.0))
    n_nonfinite = int(rec.n_nonfinite_stats.sum())
    ok = bool(R == 10 and actual_steps == int(expected_steps) and
              max_quant <= FIELD_IDENTITY_ATOL and n_nonfinite == 0)
    return dict(
        s7_pass=ok, s7_R=int(R), s7_n_record_steps=actual_steps,
        s7_expected_record_steps=int(expected_steps),
        s7_max_p_hat_times_32_integer_abs_err=max_quant,
        s7_n_nonfinite_all_stats=n_nonfinite,
        s7_criterion="R=10、記録点数がgrid通り、p_hat*32が整数、非有限統計0",
    )


# ---------------------------------------------------------------- 実行

def _format_s3_status(s3):
    """S3のconsole要約。非退化unitが無い場合のz診断は ``NA`` とする。"""
    max_z = s3.get("s3_max_abs_z")
    max_z_text = "NA" if max_z is None else f"{float(max_z):.2f}"
    return (
        f"{'PASS' if s3['s3_pass'] else 'FAIL'} "
        f"(exact err={s3['s3_max_exact_uniform_abs_err']:.2e}, "
        f"reweight err={s3['s3_max_empirical_reweighted_abs_err']:.2e}, "
        f"max|z| diagnostic={max_z_text})"
    )

def run(cfg, device, outdir, s2_steps=0):
    C, P = cfg["common"], cfg["ratchet"]
    center_alpha = float(cfg["condA"]["center_alpha"])
    selected_keys = record_key_selection(P)
    total = int(C["total_steps"])
    runs = build_runs(cfg)
    groups = group_runs(runs)
    if len(groups) != 1:
        raise ValueError(f"ratchet_log は単一グループ前提 (§3.1) だが {len(groups)} 個: "
                         f"{sorted(groups)}")
    gkey, gruns = next(iter(groups.items()))
    period = int(gruns[0]["period"])
    if any(int(r["period"]) != period for r in gruns):
        raise ValueError("グループ内 period 一様が前提 (境界整列が壊れる)")

    steps = record_steps(total, period, int(P["boundary_window"]), int(P["bulk_every"]))
    R, h, m, f = len(gruns), int(gkey[1]), int(cfg["condA"]["m"]), int(cfg["condA"]["f"])
    print(f"group={gkey}  R={R} h={h}  total={total}  記録点={len(steps)}  "
          f"境界={len(switch_steps(period, total))}", flush=True)

    # S3 の突き合わせ点 3 箇所 (§7): 序盤 / 中盤 / 末尾の記録点
    s3_steps = sorted({steps[len(steps) // 4], steps[len(steps) // 2], steps[-1]})
    rec = Recorder(steps, R, h, m, f, s3_steps=s3_steps, **selected_keys)
    t0 = time.time()
    st, elapsed = train_group(gkey, gruns, cfg, device, outdir,
                              total_steps=total, probe=rec, probe_steps=steps)
    rec.check_complete()
    print(f"  train+probe {elapsed:.1f}s  probe 呼び出し {rec.n_calls}", flush=True)

    sanity = dict(S3=check_s3(rec), S4=check_s4(rec, period), S5=check_s5(rec))
    omp_threads = os.environ.get("OMP_NUM_THREADS", "(未設定)")
    sanity["S1"] = dict(s1_pass=bool(omp_threads == "1" and torch.get_num_threads() == 1),
                        omp_num_threads=omp_threads,
                        torch_num_threads=torch.get_num_threads())
    if "mu_titration" in cfg:
        sanity["S7"] = check_s7(rec, R, expected_steps=20901)

    # --- S2: probe の無擾乱性 (§7)。同一 config を probe なしで再走させ最終状態を比較
    if s2_steps:
        print(f"  S2: probe なしで {s2_steps} step 再走 ...", flush=True)
        s2_probe_steps = record_steps(s2_steps, period, int(P["boundary_window"]),
                                      int(P["bulk_every"]))
        s2_s3_steps = sorted({s2_probe_steps[len(s2_probe_steps) // 4],
                              s2_probe_steps[len(s2_probe_steps) // 2],
                              s2_probe_steps[-1]})
        if len(s2_s3_steps) != 3:
            raise ValueError("S2 grid は S3 hook 3点を選べる長さが必要: "
                             f"steps={s2_steps}, grid={s2_probe_steps}")
        s2_rec = Recorder(s2_probe_steps, R, h, m, f, s3_steps=s2_s3_steps,
                          **selected_keys)
        st_a, _ = train_group(gkey, gruns, cfg, device, outdir, total_steps=s2_steps,
                              ckpts=[], gname="S2_with_probe",
                              probe=s2_rec, probe_steps=s2_probe_steps)
        s2_rec.check_complete()
        s2_s3 = check_s3(s2_rec)
        s2_probe_calls = int(s2_rec.n_calls)
        s2_hook_steps = sorted(int(step) for step in s2_rec.s3)
        s2_hook_calls = len(s2_hook_steps)
        s2_probe_contract_pass = bool(
            s2_probe_calls == len(s2_probe_steps) and
            s2_hook_calls == 3 and s2_hook_steps == s2_s3_steps and
            s2_s3["s3_pass"]
        )
        # 100k S2 Recorderは大きい。必要なsanityを抽出後、no-probe走の前に解放する。
        del s2_rec
        st_b, _ = train_group(gkey, gruns, cfg, device, outdir, total_steps=s2_steps,
                              ckpts=[], gname="S2_no_probe")
        ha, hb = state_hash(st_a), state_hash(st_b)
        diffs = [k for k in ha if ha[k] != hb[k]]
        ra, rb = reproducibility_hash(st_a), reproducibility_hash(st_b)
        repro_diffs = [k for k in ra if ra[k] != rb[k]]
        generator_diffs = [k for k in repro_diffs if k.startswith("generator.")]
        s2_pass = bool(not diffs and not repro_diffs and s2_probe_contract_pass)
        sanity["S2"] = dict(
            s2_pass=s2_pass, s2_steps=int(s2_steps), s2_diffs=diffs,
            s2_repro_diffs=repro_diffs, s2_generator_diffs=generator_diffs,
            s2_hash_with_probe=ha, s2_hash_no_probe=hb,
            s2_repro_hash_with_probe=ra, s2_repro_hash_no_probe=rb,
            s2_probe_recorder_complete=True,
            s2_probe_record_calls=s2_probe_calls,
            s2_probe_record_expected=len(s2_probe_steps),
            s2_s3_hook_calls=s2_hook_calls,
            s2_s3_steps=s2_hook_steps,
            s2_s3_expected_steps=s2_s3_steps,
            s2_probe_contract_pass=s2_probe_contract_pass,
            s2_s3=s2_s3,
        )
        detail = "" if s2_pass else f" diffs={diffs}, repro={repro_diffs}"
        print(f"  S2: {'PASS' if s2_pass else 'FAIL'}"
              f" (S3 hook={s2_hook_calls}/3){detail}", flush=True)

    required_checks = [sanity["S3"]["s3_pass"], sanity["S4"]["s4_pass"],
                       sanity["S5"]["s5_pass"]]
    if "mu_titration" in cfg:
        required_checks.extend([sanity["S1"]["s1_pass"], sanity["S7"]["s7_pass"]])
    if "S2" in sanity:
        required_checks.append(sanity["S2"]["s2_pass"])
    sanity["all_required_pass"] = bool(all(required_checks))

    paths = write_logs(rec, gruns, outdir, center_alpha=center_alpha)
    size_mb = sum(os.path.getsize(p) for p in paths) / 1e6
    meta = dict(elapsed_sec=round(time.time() - t0, 1), train_sec=round(elapsed, 1),
                device=device, date=time.strftime("%Y-%m-%d %H:%M:%S"),
                group=str(gkey), R=R, width=h, total_steps=total, period=period,
                alpha=center_alpha, center_alpha=center_alpha,
                n_record_steps=len(steps), n_boundaries=len(switch_steps(period, total)),
                n_realized_flips=int(sanity["S4"]["s4_n_flip_transitions"]),
                logs_mb=round(size_mb, 1), sanity=sanity,
                tracked_keys=list(rec.buf),
                step0_repro_hash=rec.step0_repro_hash,
                spec=cfg.get("spec", "specs/spec_ratchet_log_0819.md"))
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)
    print(f"  logs {size_mb:.0f} MB -> {outdir}/logs/", flush=True)
    print(f"  S3: {_format_s3_status(sanity['S3'])}", flush=True)
    print(f"  S4: {'PASS' if sanity['S4']['s4_pass'] else 'FAIL'} "
          f"(flip 遷移 {sanity['S4']['s4_n_flip_transitions']})", flush=True)
    print(f"  S5: {'PASS' if sanity['S5']['s5_pass'] else 'FAIL'} "
          f"(raw mismatch={sanity['S5']['s5_n_mismatch_raw']}, "
          f"max identity err={sanity['S5']['s5_max_pre_identity_abs_err']:.2e})", flush=True)
    if "S7" in sanity:
        print(f"  S7: {'PASS' if sanity['S7']['s7_pass'] else 'FAIL'} "
              f"(R={sanity['S7']['s7_R']}, records={sanity['S7']['s7_n_record_steps']}, "
              f"nonfinite={sanity['S7']['s7_n_nonfinite_all_stats']})", flush=True)
    return meta


def apply_smoke(cfg):
    """§4.3 のグリッド健全性スモーク: seed 1 本 / 0->50k (境界 5 個)。"""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["common"].update(total_steps=50000, seeds=[0], checkpoints=[])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ratchet_log_0819.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="seed 1 本 / 0->50k (results/_smoke_ratchet)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--s2-steps", type=int, default=0,
                    help="S2 (probe あり/なしの bit 一致) をこの step 数で実施。仕様は 100000")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.seeds is not None:
        cfg["common"]["seeds"] = list(args.seeds)
    if args.total_steps is not None:
        cfg["common"]["total_steps"] = int(args.total_steps)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)

    outdir = args.outdir or (os.path.join(ROOT, "results", "_smoke_ratchet")
                             if args.smoke else resolve_outdir(args.config))
    os.makedirs(outdir, exist_ok=True)
    print(f"outdir: {outdir}", flush=True)
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)

    run(cfg, device, outdir, s2_steps=args.s2_steps)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

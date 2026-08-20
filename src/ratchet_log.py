"""ratchet_log_0819 Phase 1: 整流モデルの時間発展ロギング走 (spec_ratchet_log_0819 §3)。

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
UNIT_KEYS = ["cos_u_mu", "p_hat", "w_norm", "b", "v", "F_self", "F_rest", "F_gate"]
RUN_VEC_KEYS = {"G": "m", "flip_state": "f"}          # [n_rec, R, dim]
RUN_SCA_KEYS = ["E_delta", "mu_norm", "ratio_mu_cov", "cos_G_mu", "G_dot_mu",
                "eval_loss_exact"]


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


def exact_record(st, as_f64=False):
    """1 記録点ぶんの 32 パターン厳密期待値 [§3.4]。全て float64 で計算。

    返り値は numpy の dict (run 配列 [R,*] / unit 行列 [R,h])。既定は保存用の float32
    だが、as_f64=True で計算精度そのままの float64 を返す (Phase 0 の 1e-10 判定用。
    float32 に丸めた値と float64 の参照実装を比べると、丸め自体で 1e-8 台の相対誤差が
    出て判定が意味を失う)。

    記号 (net の勾配は nets.VecMLP.grads と同じ係数 2 込み):
      δ = ŷ − y、G = E[δx]、µ̂ = E[x] = flip_state ‖ 0.5·1
      F_i^gate = −2η v_i E[δ·gate_i·x]、F_i^ungate = −2η v_i G
      δ = δ_self,i + δ_rest,i、δ_self,i = v_i σ(w_iᵀx+b_i)
    F の µ̂ 射影は**単位ベクトル** µ̂/‖µ̂‖ に対して取る (‖µ̂‖ は flip の 1 の個数で
    時間変動するため、そのまま内積すると self/rest 系列が境界で不連続に見える)。
    ‖µ̂‖ は mu_norm として保存するので生の内積は事後復元できる。"""
    env, net, teacher = st["env"], st["net"], st["teacher"]
    with torch.no_grad():
        X = full_support_ro(env).double()                     # [P,R,m]
        y = teacher_f64(teacher, X)                           # [P,R]

        # enc=std のみを対象とする (§3.1)。centered なら x_in が running_mean に依存し、
        # µ̂ = E[x] の解析形も変わるので、黙って通さず落とす。
        cmask = st["centered"][:, None].double()              # [R,1]
        if bool(st["centered"].any()):
            raise NotImplementedError("ratchet_log は enc=std 専用 (§3.1)")
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
        mu = x_in.mean(dim=0)                                 # [R,m] = flip ‖ 0.5·1
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

        dt = np.float64 if as_f64 else np.float32
        cv = lambda t: t.detach().cpu().numpy().astype(dt)
        return dict(
            G=cv(G), flip_state=cv(env.flip_state.double()),
            E_delta=cv(E_delta), mu_norm=cv(mu_norm), ratio_mu_cov=cv(ratio),
            cos_G_mu=cv(cos_G_mu), G_dot_mu=cv(G_dot_mu),
            eval_loss_exact=cv((delta ** 2).mean(dim=0)),
            cos_u_mu=cv(cos_u_mu), p_hat=cv(p_hat), w_norm=cv(w_norm),
            b=cv(b), v=cv(v), F_self=cv(F_self), F_rest=cv(F_rest), F_gate=cv(F_gate))


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


class Recorder:
    """probe(st, step) として train_group に渡す読み取り専用アキュムレータ。

    記録点は事前に確定しているので配列を先に確保する (np.stack の一時 2 倍を避ける)。
    s3_steps に指定した記録点では S3 用に eval バッチの経験 p̂ も控える (§7)。
    eval_batch は generator を消費せず env.t も進めないので probe から呼んでよい
    (train_group の docstring 参照)。"""

    def __init__(self, steps, R, h, m, f, s3_steps=()):
        self.steps = np.asarray(steps, dtype=np.int64)
        self.index = {int(s): i for i, s in enumerate(self.steps)}
        n = len(self.steps)
        self.buf = {k: np.zeros((n, R, h), dtype=np.float32) for k in UNIT_KEYS}
        self.buf["G"] = np.zeros((n, R, m), dtype=np.float32)
        self.buf["flip_state"] = np.zeros((n, R, f), dtype=np.float32)
        for k in RUN_SCA_KEYS:
            self.buf[k] = np.zeros((n, R), dtype=np.float32)
        self.filled = np.zeros(n, dtype=bool)
        self.n_calls = 0
        self.s3_steps = set(int(s) for s in s3_steps)
        self.s3 = {}                        # step -> (p_exact [R,h], p_emp [R,h], N)

    def __call__(self, st, step):
        i = self.index.get(int(step))
        if i is None:                       # probe_steps と一致するはずだが保険
            return
        rec = exact_record(st)
        for k, arr in self.buf.items():
            arr[i] = rec[k]
        self.filled[i] = True
        self.n_calls += 1
        if int(step) in self.s3_steps:
            self.s3[int(step)] = (rec["p_hat"], *_empirical_p_hat(st))

    def check_complete(self):
        miss = np.flatnonzero(~self.filled)
        if len(miss):
            raise RuntimeError(f"未記録の grid 点が {len(miss)} 個: "
                               f"{self.steps[miss][:10]} ...")


def write_logs(rec, runs, outdir):
    """seed ごとに logs/seed{k}.npz を書く [§3.4]。"""
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    paths = []
    for i, r in enumerate(runs):
        out = dict(step=rec.steps, run_id=np.array(r["run_id"]), seed=np.int64(r["seed"]),
                   lr=np.float32(r["lr"]), period=np.int64(r["period"]),
                   width=np.int64(r["width"]))
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


def _binom_tail_ge(n, k, p):
    """P(X >= k), X ~ Binomial(n, p)。k は小さい前提で下側から補数を取る。"""
    from math import comb
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lower = sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k))
    return max(0.0, 1.0 - lower)


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


def check_s3(rec):
    """S3: 厳密 p̂ と eval_batch=2000 経験値の突き合わせ、記録点 3 箇所 [§7]。

    eval バッチのランダムビット部は iid uniform bits なので、各サンプルは 32 パターンの
    一様抽選そのもの。よって経験値は厳密に Binomial(N, p_exact)/N であり、
    z = (p̂_emp − p_exact)/√(p(1−p)/N) は N(0,1) に従う。

    **判定統計量の逸脱 (記録)**: 仕様 §7 の字義は「±3σ 内」だが、これを全ユニットの
    max に適用すると R·h·3 ≈ 3000 検定になり、真に無擾乱でも P(max|z|>3) ≈ 1−0.9973³⁰⁰⁰
    ≈ 99.98% で必ず落ちる (実際スモークで max|z|=3.15)。bias_margin_0814 の教訓
    (「相対一致でなく二項ゆらぎ基準の z 検定に置き換えるのが正しい」) に倣い、
    **median|z| ≤ 1.0 かつ |z|>3 の割合 ≤ 1%** を PASS とする。参考として max|z| も出す。
    N(0,1) なら median|z| = 0.674、|z|>3 の割合 = 0.27%。

    p_exact ∈ {0,1} のユニットは σ=0 (eval の全サンプルが 32 パターン内にあるので
    経験値も決定的に 0/1) なので、z ではなく厳密一致を要求する。"""
    per, zs, degen_err, n_deg, N = [], [], 0.0, 0, 0
    for step in sorted(rec.s3):
        p_ex, p_emp, N = rec.s3[step]
        sd = np.sqrt(np.maximum(p_ex * (1 - p_ex), 0) / N)
        degen = sd == 0
        z = np.where(degen, 0.0, (p_emp - p_ex) / np.maximum(sd, 1e-12))[~degen]
        e = float(np.abs(p_emp[degen] - p_ex[degen]).max()) if degen.any() else 0.0
        zs.append(z)
        degen_err, n_deg = max(degen_err, e), n_deg + int(degen.sum())
        per.append(dict(step=int(step), n_z=int(z.size),
                        median_abs_z=round(float(np.median(np.abs(z))), 4)
                        if z.size else None,
                        max_abs_z=round(float(np.abs(z).max()), 4) if z.size else None,
                        frac_gt3=round(float((np.abs(z) > 3).mean()), 5) if z.size else None,
                        n_degenerate=int(degen.sum()), max_degenerate_err=e))
    z = np.concatenate(zs) if zs else np.zeros(0)
    med = float(np.median(np.abs(z))) if z.size else 0.0
    n, k = int(z.size), int((np.abs(z) > 3).sum())
    tail = _binom_tail_ge(n, k, 0.0026998)          # P(X>=k), X~Bin(n, P(|Z|>3))
    ok = bool(med <= 1.0 and tail >= 0.001 and degen_err == 0.0)
    return dict(s3_pass=ok, s3_median_abs_z=round(med, 4),
                s3_n_gt3=k, s3_expected_gt3=round(0.0026998 * n, 3),
                s3_binom_tail_p=round(tail, 5),
                s3_max_abs_z=round(float(np.abs(z).max()), 4) if z.size else None,
                s3_n_z=n, s3_n_degenerate=n_deg,
                s3_max_degenerate_err=degen_err, s3_eval_N=int(N),
                s3_criterion="median|z|<=1.0 かつ |z|>3 の個数の二項上側 p>=0.001 かつ "
                             "退化ユニット厳密一致 (§7 字義の max ±3σ からの逸脱)",
                s3_note="eval バッチは固定なので、ゲート集合が安定なユニットの z は "
                        "記録点をまたいで同値になる (独立でない)。二項検定は n を "
                        "水増しする向き = PASS しやすい向きに保守的でないが、"
                        "median|z| 側が主判定なので許容する。",
                s3_per_step=per)


def check_s4(rec, period):
    """S4: flip_state が変化する step が t ≡ 0 (mod period) の直後だけであること [§7]。

    probe は train_group のループ本体先頭 (env.step() の前) で呼ばれるので、
    境界 B の flip は「記録点 B」と「記録点 B+1」の間で起きる。したがって
    flip_state が動く記録点ペアの左端は必ず period の倍数になる。"""
    fs = rec.buf["flip_state"]                                     # [n,R,f]
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


# ---------------------------------------------------------------- 実行

def run(cfg, device, outdir, s2_steps=0):
    C, P = cfg["common"], cfg["ratchet"]
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
    rec = Recorder(steps, R, h, m, f, s3_steps=s3_steps)
    t0 = time.time()
    st, elapsed = train_group(gkey, gruns, cfg, device, outdir,
                              total_steps=total, probe=rec, probe_steps=steps)
    rec.check_complete()
    print(f"  train+probe {elapsed:.1f}s  probe 呼び出し {rec.n_calls}", flush=True)

    sanity = dict(S3=check_s3(rec), S4=check_s4(rec, period))
    sanity["S1"] = dict(omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                        torch_num_threads=torch.get_num_threads())

    # --- S2: probe の無擾乱性 (§7)。同一 config を probe なしで再走させ最終状態を比較
    if s2_steps:
        print(f"  S2: probe なしで {s2_steps} step 再走 ...", flush=True)
        st_a, _ = train_group(gkey, gruns, cfg, device, outdir, total_steps=s2_steps,
                              ckpts=[], gname="S2_with_probe",
                              probe=Recorder(record_steps(s2_steps, period,
                                                          int(P["boundary_window"]),
                                                          int(P["bulk_every"])),
                                             R, h, m, f),
                              probe_steps=record_steps(s2_steps, period,
                                                       int(P["boundary_window"]),
                                                       int(P["bulk_every"])))
        st_b, _ = train_group(gkey, gruns, cfg, device, outdir, total_steps=s2_steps,
                              ckpts=[], gname="S2_no_probe")
        ha, hb = state_hash(st_a), state_hash(st_b)
        diffs = [k for k in ha if ha[k] != hb[k]]
        sanity["S2"] = dict(s2_pass=not diffs, s2_steps=int(s2_steps), s2_diffs=diffs,
                            s2_hash_with_probe=ha, s2_hash_no_probe=hb)
        print(f"  S2: {'PASS' if not diffs else 'FAIL ' + str(diffs)}", flush=True)

    paths = write_logs(rec, gruns, outdir)
    size_mb = sum(os.path.getsize(p) for p in paths) / 1e6
    meta = dict(elapsed_sec=round(time.time() - t0, 1), train_sec=round(elapsed, 1),
                device=device, date=time.strftime("%Y-%m-%d %H:%M:%S"),
                group=str(gkey), R=R, width=h, total_steps=total, period=period,
                n_record_steps=len(steps), n_boundaries=len(switch_steps(period, total)),
                n_realized_flips=int(sanity["S4"]["s4_n_flip_transitions"]),
                logs_mb=round(size_mb, 1), sanity=sanity,
                spec="specs/spec_ratchet_log_0819.md")
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)
    print(f"  logs {size_mb:.0f} MB -> {outdir}/logs/", flush=True)
    print(f"  S3: {'PASS' if sanity['S3']['s3_pass'] else 'FAIL'} "
          f"(max|z|={sanity['S3']['s3_max_abs_z']:.2f})", flush=True)
    print(f"  S4: {'PASS' if sanity['S4']['s4_pass'] else 'FAIL'} "
          f"(flip 遷移 {sanity['S4']['s4_n_flip_transitions']})", flush=True)
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

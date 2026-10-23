#!/usr/bin/env python3
"""chargpt_detlabel_1023 の学習の wrapper（spec specs/spec_chargpt_detlabel_1023.md §4.5・§6.3・§6.4・§8・§10.1・§12）。

土台（書き換えない。このブランチの src/ に blob のまま写した。写し元はブランチ claude/chargpt_cl_1019 の 033679e0）:
  src/chargpt_cl_1019.py       blob cc524f85d5f2d9813a272abb6b9476b065f1c17b（RUN_ID = "chargpt_cl_1019" のまま）
  src/chargpt_cl_1019_kune.py  blob 272e75548fef16d364c464327c8e44d034b9cbb2
kune.py と同じ作りで、import したうえで部品を差し替える。差し替えるもの:
  1. C.stationary: 先に計算した π（rule_sha で引く）があればそれを返す（§10.3 (i)）。無ければ元の関数で計算する。
     π は元の ChainM で計算した値をそのまま置いたもの（analysis/chargpt_detlabel_1023/W/pi_precompute.py）。S-pi で照合する。
  2. C.task_cpu_M: 元の関数のあとで、課題ごとの連鎖の記録 (r3)・π (r4) と S-chain・S-entropy（§8）を裏のスレッドで出す。
     外れたら CHECK_FAILED_chain_t<t>.json を書いて例外を投げ、走を止める（--checks_mode stop・--entropy_check stop が既定）。
     S-entropy だけを記録にする引数 --entropy_check record を分けて置いた（spec の規則のままでは α ≤ 1e−3 の連鎖の
     3 割ほどで外れる。checks/W の報告を見よ。どちらを使うかは主・Issa が決める。既定は spec のとおり止める）。
  3. C.run_task: 本体の課題（measure_cb がある課題）に (r1)(r2)(r5) の記録器を足す。記録の時点は元の測定点に足すだけで、
     元の測定（measure_cb）は元の時点でだけ呼ぶ。記録は GPU の上の写しと no_grad の forward だけで、学習の軌道に触らない（S-recorder）。
  4. C.Engine._step_body: --target soft（腕 S2、§4.5）のときだけ、損失を真の分布との交差エントロピー −Σ_c p(c|a,b) log q(c) にする。
     online.npz の loss はサンプルした目標の交差エントロピー（A2 と同じ定義、同じ logits から no_grad で計算）、
     online.npz の aux は勾配に使った真の分布との交差エントロピー（A2・A4・A3 では aux = loss）。fresh の網も同じ損失で学ぶ。
  5. C.run: 走の後に within_motion.npz・logit_stats.csv・pi.npz を書き、provenance.json に detlabel の項目を足す。
     完走の印は provenance.json の detlabel_complete = true。
  6. replay サブコマンド（§6.4 の A2 の再生の入口）: 課題 t−1 末の checkpoint（重みと Adam）から課題 t を同じデータ列で回し、
     (r1)(r2) を記録し、A2 の online.npz・task<t>.pt との差を記録する。

使い方（本走。腕は --alpha と --target で決める: A4 = 1e-4 sample、A3 = 1e-3 sample、S2 = 0.01 soft）:
  CHARGPT_NLAYER=2 CHARGPT_DECOMP_CKPT=1 python src/chargpt_detlabel_1023.py run --box M --arm C3 --seed 0 --tasks 100 \
      --n_per_task 10000 --alpha 1e-4 --target sample --fresh_tasks 10 25 50 100 \
      --pi_dir results/chargpt_detlabel_1023/pi_cache --out results/chargpt_detlabel_1023/M2/A4/seed0
  再生:
  CHARGPT_NLAYER=2 python src/chargpt_detlabel_1023.py replay --seed 0 --tasks 2 3 --alpha 0.01 \
      --src <A2 の seed0 の dir> --pi_dir results/chargpt_detlabel_1023/pi_cache --out <dir>
"""
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019_kune as K  # noqa: E402  （CHARGPT_NLAYER を読み、C を差し替える）

C = K.C
V = C.V
LOG65 = math.log(65)
B_MAX = 2.5e-4                      # §8 S-chain: B = 64·d·log 65 ≤ δ/100
ROWSUM_MAX = 1e-12
MOTION_Q = ("net5k", "path5k", "net100", "rot5k", "norm5k")
CHAIN_KEYS = ("task", "alpha", "H", "pi_iters", "pi_src", "pi_last_step_L1", "B", "rowsum_err", "nan",
              "zero_drawn_data", "zero_drawn_probe", "zero_drawn_eval",
              "label_nonmode_frac_all", "label_nonmode_frac_16k", "distinct_ctx_all", "distinct_ctx_500upd",
              "copy_correct_frac_16k", "copy_correct_frac_16k_second_half",
              "true_ce_eval", "true_ce_eval_se", "true_ce_eval_minus_H", "true_ce_data",
              "true_ce_eval_se_seq", "true_ce_eval_se_truevar", "ok_entropy_seq_or_truevar",
              "ok_chain", "ok_entropy", "cpu_task_sec", "rec_sec", "rule_sha", "pi_sha")

_DL = dict(target="sample", recorder=True, chain_rec=False, checks_mode="stop", entropy_check="stop", n=None, out=None,
           motion={}, alive={}, logit_rows=[], chain_rows={}, pis={}, pi_src={}, lock=threading.Lock())
_PI = {}                             # rule_sha -> dict(pi, iters, src, pi_sha)
_LOADED = set()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def alpha_tag(alpha):
    return f"a{alpha:g}"


# ---------------------------------------------------------------- 1. π の読み込み
def load_pi_dir(pi_dir, alpha, seed):
    d = Path(pi_dir) / alpha_tag(alpha) / f"seed{seed}"
    n = 0
    if not d.exists():
        return 0
    for f in sorted(d.glob("t*.npz")):
        if f"file:{f}" in _LOADED:
            continue
        _LOADED.add(f"file:{f}")
        z = np.load(f, allow_pickle=False)
        pi = z["pi"]
        if sha256_bytes(pi.tobytes()) != str(z["pi_sha"]):
            raise SystemExit(f"π の sha256 が合わない: {f}")
        _PI[str(z["rule_sha"])] = dict(pi=pi, iters=int(z["pi_iters"]), src=f"file:{f}", pi_sha=str(z["pi_sha"]))
        n += 1
    return n


_orig_stationary = C.stationary


def stationary_cached(P):
    sha = sha256_bytes(P.tobytes())             # = rule_sha（ChainM は s.P を渡す。task_cpu_M の rule_sha と同じ bytes）
    e = _PI.get(sha)
    if e is None and _DL.get("pi_key"):        # 走の途中で先計算が置いた file を読み直す
        load_pi_dir(*_DL["pi_key"])
        e = _PI.get(sha)
    if e is None:
        pi, it = _orig_stationary(P)
        with _DL["lock"]:
            _DL["pi_src"][sha] = "computed"
        return pi, it
    with _DL["lock"]:
        _DL["pi_src"][sha] = e["src"]
    return e["pi"].copy(), e["iters"]


C.stationary = stationary_cached


# ---------------------------------------------------------------- 2. 連鎖の記録 (r3)(r4) と S-chain・S-entropy
def copy_hits(ctx, tg):
    """prereg/chain_stats.py と同じ定義: 同じ系列で同じ文脈の前回の続きを写して当たる目標。"""
    n = ctx.shape[0]
    last = np.full((n, V * V), -1, np.int8)
    corr = np.zeros_like(tg, bool)
    ar = np.arange(n)
    for j in range(ctx.shape[1]):
        prev = last[ar, ctx[:, j]]
        corr[:, j] = prev == tg[:, j]
        last[ar, ctx[:, j]] = tg[:, j]
    return corr


def seq_ctx(x):
    x = x.astype(np.int64)
    return x[:, :-2] * V + x[:, 1:-1], x[:, 2:]


def chain_record(c, seed, task, alpha):
    t0 = time.time()
    ch = c["chain"]
    P, pi = ch.P, ch.pi
    P2 = P.reshape(V * V, V)
    nan = bool(np.isnan(P).any() or np.isnan(pi).any())
    new = 0.5 * pi + 0.5 * np.einsum("ab,abc->bc", pi, P)          # prereg/chain_stats.py の pi_last_step_L1 と同じ定義
    d = float(np.abs(new - pi).sum())
    B = 64.0 * d * LOG65
    mode = P2.argmax(1)
    data = c["data"].reshape(-1, C.M_SEQ)
    nz = nonmode = ntok = 0
    seen = np.zeros(V * V, bool)
    lp_sum = 0.0
    for i0 in range(0, data.shape[0], 32000):
        cx, tg = seq_ctx(data[i0:i0 + 32000].numpy())
        p = P2[cx, tg]
        nz += int((p == 0).sum())
        nonmode += int((tg != mode[cx]).sum())
        ntok += tg.size
        seen[cx.ravel()] = True
        lp_sum += float(-np.log(np.maximum(p, 1e-300)).sum())
    cx16, tg16 = seq_ctx(data[:16000].numpy())
    corr = copy_hits(cx16, tg16)
    cxp, tgp = seq_ctx(c["probe"].numpy())                         # 鎖 probe は 128 トークン（目標 126）
    cxe, tge = seq_ctx(c["eval_seq"].numpy())
    pe = P2[cxe, tge]
    lp = -np.log(np.maximum(pe, 1e-300))
    ce, se = float(lp.mean()), float(lp.std(ddof=1) / math.sqrt(lp.size))
    se_seq = float(lp.mean(1).std(ddof=1) / math.sqrt(lp.shape[0]))      # 記録: 系列ごとの平均から（系列の中の相関を含む）
    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = np.where(P2 > 0, P2 * np.log(np.where(P2 > 0, P2, 1.0)) ** 2, 0.0).sum(1)
    se_true = math.sqrt(max(float((pi.reshape(-1) * m2).sum() - ch.H ** 2), 0.0) / lp.size)   # 記録: 連鎖の真の分散から
    zp, ze = int((P2[cxp, tgp] == 0).sum()), int((pe == 0).sum())
    rule_sha = c["rule_sha"]
    with _DL["lock"]:
        src = _DL["pi_src"].get(rule_sha, "unknown")
    row = dict(task=task, alpha=alpha, H=ch.H, pi_iters=int(ch.pi_iters), pi_src=src, pi_last_step_L1=d, B=B,
               rowsum_err=ch.rowsum_err, nan=nan, zero_drawn_data=nz, zero_drawn_probe=zp, zero_drawn_eval=ze,
               label_nonmode_frac_all=nonmode / ntok, label_nonmode_frac_16k=float((tg16 != mode[cx16]).mean()),
               distinct_ctx_all=int(seen.sum()), distinct_ctx_500upd=int(np.unique(cx16).size),
               copy_correct_frac_16k=float(corr.mean()), copy_correct_frac_16k_second_half=float(corr[:, 63:].mean()),
               true_ce_eval=ce, true_ce_eval_se=se, true_ce_eval_minus_H=ce - ch.H, true_ce_data=lp_sum / ntok,
               rule_sha=rule_sha, pi_sha=sha256_bytes(pi.tobytes()))
    row["ok_chain"] = bool(ch.rowsum_err <= ROWSUM_MAX and not nan and nz + zp + ze == 0 and B <= B_MAX)
    row["ok_entropy"] = bool(abs(ce - ch.H) <= 4.0 * se)                 # spec §8 の S-entropy（トークンごとの標準誤差）
    row["true_ce_eval_se_seq"], row["true_ce_eval_se_truevar"] = se_seq, se_true
    row["ok_entropy_seq_or_truevar"] = bool(abs(ce - ch.H) <= 4.0 * max(se_seq, se_true))   # 記録だけ（代案。判定に使わない）
    row["cpu_task_sec"] = c.get("cpu_task_sec", float("nan"))
    row["rec_sec"] = time.time() - t0
    return row


_orig_task_cpu_M = C.task_cpu_M


def task_cpu_M_rec(seed, task, n, alpha):
    t0 = time.time()
    c = _orig_task_cpu_M(seed, task, n, alpha)
    c["cpu_task_sec"] = time.time() - t0
    if _DL["chain_rec"] and n == _DL["n"]:
        row = chain_record(c, seed, task, alpha)
        with _DL["lock"]:
            _DL["chain_rows"][task] = row
            _DL["pis"][task] = c["chain"].pi.copy()
        out = _DL["out"]
        if out is not None:
            f = out / "chain_stats.csv"
            new = not f.exists()
            with open(f, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(CHAIN_KEYS))
                if new:
                    w.writeheader()
                w.writerow({k: (f"{row[k]:.9g}" if isinstance(row[k], float) else row[k]) for k in CHAIN_KEYS})
        if not (row["ok_chain"] and row["ok_entropy"]):
            if out is not None:
                (out / f"CHECK_FAILED_chain_t{task}.json").write_text(json.dumps(row, indent=1))
            msg = f"S-chain/S-entropy に外れた: seed {seed} 課題 {task} ok_chain={row['ok_chain']} ok_entropy={row['ok_entropy']}"
            stop_chain = not row["ok_chain"] and _DL["checks_mode"] == "stop"
            stop_ent = not row["ok_entropy"] and _DL["checks_mode"] == "stop" and _DL["entropy_check"] == "stop"
            if stop_chain or stop_ent:
                raise RuntimeError(msg)
            print("[warn] " + msg, flush=True)
    return c


C.task_cpu_M = task_cpu_M_rec


# ---------------------------------------------------------------- 3. 記録器 (r1)(r2)(r5)
class MotionRec:
    """課題の中の up 行の動き (r1)、生きた unit (r2)、課題末の logit の幅 (r5)。GPU の上の写しと no_grad の forward だけ。"""

    def __init__(s, model, ctx, task):
        s.m, s.ctx, s.task, n = model, ctx, task, ctx.n
        s.n, s.u100, s.half = n, min(100, n), n // 2
        step = max(1, (n - s.half) // 10)
        s.seg = [s.half + k * step for k in range(1, 11)]
        s.seg[-1] = n
        s.points = {s.u100, s.half, *s.seg}
        s.path = None
        s.alive = np.zeros((C.L, C.FF, 2), bool)

    def _w(s):
        return [blk.mlp.up.weight.detach().double().clone() for blk in s.m.h]

    @torch.no_grad()
    def _alive(s):
        cap = []
        s.m.body(s.ctx.probe, cap)
        return np.stack([(cap[2 * li + 1].reshape(-1, C.FF) > 0).any(0).cpu().numpy() for li in range(C.L)])

    @torch.no_grad()
    def at(s, u):
        if u not in s.points:
            return
        W = s._w()
        if u == s.u100:
            s.w100 = W
        if u == s.half:
            s.wh, s.prev = W, W
            s.path = [torch.zeros(C.FF, dtype=torch.float64, device=W[0].device) for _ in W]
            s.alive[:, :, 0] = s._alive()
        if u in s.seg and u > s.half:
            for li in range(len(W)):
                s.path[li] += (W[li] - s.prev[li]).norm(dim=1)
            s.prev = W
        if u == s.n:
            q = np.zeros((C.L, C.FF, len(MOTION_Q)), np.float32)
            for li in range(len(W)):
                a, b = s.wh[li], W[li]
                cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-300)
                q[li, :, 0] = (b - a).norm(dim=1).float().cpu().numpy()
                q[li, :, 1] = s.path[li].float().cpu().numpy()
                q[li, :, 2] = (b - s.w100[li]).norm(dim=1).float().cpu().numpy()
                q[li, :, 3] = (1.0 - cos).float().cpu().numpy()
                q[li, :, 4] = a.norm(dim=1).float().cpu().numpy()
            s.alive[:, :, 1] = s._alive()
            s.q = q
            s.logit = s._logit_stats()

    @torch.no_grad()
    def _logit_stats(s):
        lg = s.m(s.ctx.probe).double()                          # (64, 128, 65)
        top = lg.topk(2, dim=-1).values
        pm = lg.softmax(-1).max(-1).values
        return dict(task=s.task, gap12_med=float((top[..., 0] - top[..., 1]).median()), maxprob_mean=float(pm.mean()),
                    gamma_f_norm=float(s.m.nf.weight.double().norm()),
                    emb_rownorm_med=float(s.m.wte.weight.double().norm(dim=1).median()))


_orig_run_task = C.run_task                  # = kune.run_task_freeze


def run_task_dl(model, eng, ctx, *, warmup_from, measure_cb=None, points=(), **kw):
    if _DL["target"] == "soft":
        load_soft_P(ctx)
    if measure_cb is None or not _DL["recorder"]:          # fresh の網（measure_cb なし）は記録しない
        return _orig_run_task(model, eng, ctx, warmup_from=warmup_from, measure_cb=measure_cb, points=points, **kw)
    t = warmup_from // ctx.n + 1
    rec = MotionRec(model, ctx, t)
    orig_pts = set(points)
    inner = measure_cb

    def cb(u):
        rec.at(u)
        if u in orig_pts:
            inner(u)

    res = _orig_run_task(model, eng, ctx, warmup_from=warmup_from, measure_cb=cb, points=orig_pts | rec.points, **kw)
    _DL["motion"][t], _DL["alive"][t] = rec.q, rec.alive
    _DL["logit_rows"].append(rec.logit)
    return res


C.run_task = run_task_dl


# ---------------------------------------------------------------- 4. S2 の損失（真の分布を目標に）
_SOFT = {}


def load_soft_P(ctx):
    P = torch.from_numpy(ctx.chain.P.reshape(V * V, V)).to(torch.float32)
    _SOFT["P"].copy_(P)                       # CUDA グラフの外で静的な tensor に写す
    _SOFT["task_rule_sha"] = ctx.rule_sha


def soft_losses(out, x, y):
    lg = out[:, 1:].reshape(-1, V)
    ci = (x[:, :-1] * V + x[:, 1:]).reshape(-1)            # 目標 y[:, j] = 系列の j+1 番目、文脈 (x[:, j−1], x[:, j])
    soft = F.cross_entropy(lg, _SOFT["P"].index_select(0, ci))
    with torch.no_grad():
        samp = F.cross_entropy(lg.detach(), y[:, 1:].reshape(-1))     # A2 の Engine._loss と同じ式
    return soft, samp


_prev_step_body = C.Engine._step_body        # = kune._step_body_cap


def _step_body_dl(s):
    if _DL["target"] != "soft":
        return _prev_step_body(s)
    assert not any(getattr(b.mlp, "capped", False) for b in s.m.h), "S2 は cap の腕に対応しない"
    soft, samp = soft_losses(s.m(s.x), s.x, s.y)
    soft.backward()
    s.opt.step()
    return samp.detach(), soft.detach()      # buf_loss = サンプルの CE（A2 と同じ定義）、buf_aux = 勾配に使った CE


C.Engine._step_body = _step_body_dl


# ---------------------------------------------------------------- 5. 走の後の保存と provenance
def file_info():
    here = Path(__file__).resolve().parent
    out = {}
    for f in ("chargpt_cl_1019.py", "chargpt_cl_1019_kune.py", "chargpt_detlabel_1023.py"):
        b = (here / f).read_bytes()
        out[f] = dict(sha256=sha256_bytes(b), git_blob=hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest())
    return out


def save_records(out, n_tasks):
    if _DL["motion"]:
        mot = np.full((n_tasks, C.L, C.FF, len(MOTION_Q)), np.nan, np.float32)
        alv = np.zeros((n_tasks, C.L, C.FF, 2), bool)
        have = np.zeros(n_tasks, bool)
        for t, q in _DL["motion"].items():
            mot[t - 1], alv[t - 1], have[t - 1] = q, _DL["alive"][t], True
        np.savez_compressed(out / "within_motion.npz", motion=mot, alive=alv, have=have, quantities=np.array(MOTION_Q),
                            alive_points=np.array(["u_half", "u_end"]))
    if _DL["logit_rows"]:
        with open(out / "logit_stats.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(_DL["logit_rows"][0]))
            w.writeheader()
            for r in sorted(_DL["logit_rows"], key=lambda r: r["task"]):
                w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    if _DL["pis"]:
        ts = sorted(_DL["pis"])
        np.savez_compressed(out / "pi.npz", tasks=np.array(ts), pi=np.stack([_DL["pis"][t] for t in ts]),
                            rule_sha=np.array([_DL["chain_rows"][t]["rule_sha"] for t in ts]),
                            pi_sha=np.array([_DL["chain_rows"][t]["pi_sha"] for t in ts]))


_prev_run = C.run                            # = kune.run_wrap


def run_dl(args):
    if os.environ.get("CHARGPT_DECOMP_CKPT") != "1" and args.ckpt:
        raise SystemExit("CHARGPT_DECOMP_CKPT=1 で起動する（§6.2）")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in [out / "chain_stats.csv", *out.glob("CHECK_FAILED_chain_t*.json")]:
        f.unlink(missing_ok=True)            # 前の半端な走の記録を消す（完走の記録は provenance.json の印で守る）
    pv = out / "provenance.json"
    pv.unlink(missing_ok=True)
    _DL.update(target=args.target, recorder=not args.no_recorder, chain_rec=args.box == "M", checks_mode=args.checks_mode,
               entropy_check=args.entropy_check,
               n=args.n_per_task, out=out)
    n_pi = load_pi_dir(args.pi_dir, args.alpha, args.seed) if args.pi_dir else 0
    _DL["pi_key"] = (args.pi_dir, args.alpha, args.seed) if args.pi_dir else None
    if args.target == "soft":
        _SOFT["P"] = torch.zeros(V * V, V, dtype=torch.float32, device=args.device)
    t0 = time.time()
    _prev_run(args)
    save_records(out, args.tasks)
    rows = [_DL["chain_rows"][t] for t in sorted(_DL["chain_rows"])]
    d = json.loads(pv.read_text())
    d.update(detlabel=dict(
        spec="specs/spec_chargpt_detlabel_1023.md", target=args.target, recorder=not args.no_recorder,
        online_aux_meaning=("CE against the true conditional distribution (gradient loss)" if args.target == "soft"
                            else "same as loss (sample CE)"),
        pi_dir=args.pi_dir, pi_files_loaded=n_pi,
        pi_source_counts={k: sum(1 for r in rows if r["pi_src"].split(":")[0] == k) for k in ("file", "computed", "unknown")},
        gpu_concurrency=int(os.environ.get("DETLABEL_GPU_CONCURRENCY", "1")),
        pi_mode=("precomputed" if n_pi else "in-run") + f"+gpu_parallel{os.environ.get('DETLABEL_GPU_CONCURRENCY', '1')}",
        checks_mode=args.checks_mode, entropy_check=args.entropy_check,
        chain_fail_tasks_S_chain=[r["task"] for r in rows if not r["ok_chain"]],
        chain_fail_tasks_S_entropy=[r["task"] for r in rows if not r["ok_entropy"]],
        chain_fail_tasks=[r["task"] for r in rows if not (r["ok_chain"] and r["ok_entropy"])],
        chain_B_max=max((r["B"] for r in rows), default=None), files=file_info(),
        wall_sec_wrapper=time.time() - t0, motion_quantities=list(MOTION_Q)))
    d["detlabel_complete"] = True
    pv.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    return out


C.run = run_dl


# ---------------------------------------------------------------- 6. A2 の再生の入口（§6.4）
def replay(a):
    assert C.L == 2, "CHARGPT_NLAYER=2 で起動する"
    C.setup_torch()
    dev = torch.device(a.device)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _DL.update(target="sample", recorder=True, chain_rec=False, n=a.n_per_task, out=None)
    n_pi = load_pi_dir(a.pi_dir, a.alpha, a.seed) if a.pi_dir else 0
    _DL["pi_key"] = (a.pi_dir, a.alpha, a.seed) if a.pi_dir else None
    model = C.build_model("C3", "M", a.seed, dev)
    eng = C.Engine(model, "C3", "M", graph=a.engine == "graph")
    params = list(model.parameters())
    a2_online = np.load(src / "online.npz")["loss"]
    rows = []
    for t in a.tasks:
        tt = time.time()
        sd = torch.load(src / "ckpts" / f"task{t - 1}.pt", map_location=dev)
        model.load_state_dict(sd)                              # 記憶の場所を変えずに写す（グラフが指す tensor のまま）
        asd = torch.load(src / "ckpts" / f"task{t - 1}_adam.pt", map_location=dev)
        assert len(asd["state"]) == len(params)
        with torch.no_grad():
            for i, p in enumerate(params):
                st, sv = eng.opt.state[p], asd["state"][i]
                for k in ("exp_avg", "exp_avg_sq", "step"):
                    st[k].copy_(sv[k].to(st[k].dtype))
        cpu = C.task_cpu_M(a.seed, t, a.n_per_task, a.alpha)
        ctx = C.TaskCtx("M", a.seed, t, a.n_per_task, dev, alpha=a.alpha, cpu=cpu)
        loss, aux = C.run_task(model, eng, ctx, warmup_from=(t - 1) * a.n_per_task, measure_cb=lambda u: None, points=())
        torch.cuda.synchronize()
        ref = a2_online[t - 1]
        dl = np.abs(loss.astype(np.float64) - ref.astype(np.float64))
        end_ref = torch.load(src / "ckpts" / f"task{t}.pt", map_location=dev)
        cur = model.state_dict()
        rel = {k: float((cur[k].double() - v.double()).abs().max() / v.double().abs().max().clamp_min(1e-300))
               for k, v in end_ref.items()}
        bit_state = all(torch.equal(cur[k], v) for k, v in end_ref.items())
        np.savez_compressed(out / f"replay_task{t}.npz", motion=_DL["motion"][t], alive=_DL["alive"][t], online=loss,
                            quantities=np.array(MOTION_Q), alive_points=np.array(["u_half", "u_end"]))
        rows.append(dict(seed=a.seed, task=t, online_bitwise=bool(np.array_equal(loss, ref)), online_absdiff_max=float(dl.max()),
                         online_absdiff_mean=float(dl.mean()), state_bitwise=bit_state, state_reldiff_max=max(rel.values()),
                         state_reldiff_argmax=max(rel, key=rel.get), data_sha=ctx.data_sha, rule_sha=ctx.rule_sha,
                         pi_src=_DL["pi_src"].get(ctx.rule_sha, "unknown"), sec=time.time() - tt))
        print(f"[replay s{a.seed}] task {t} online_bitwise {rows[-1]['online_bitwise']} max|Δ| {dl.max():.3g} "
              f"state_bitwise {bit_state} rel {rows[-1]['state_reldiff_max']:.3g} {rows[-1]['sec']:.0f}s", flush=True)
        del ctx, cpu
    with open(out / "replay_compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    import socket
    (out / "replay_provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, host=socket.gethostname(), device=torch.cuda.get_device_name(dev), torch=torch.__version__,
        src=str(src), seed=a.seed, tasks=a.tasks, alpha=a.alpha, n_per_task=a.n_per_task, pi_files_loaded=n_pi,
        files=file_info(), n_layer=C.L, complete=True), indent=1))


# ---------------------------------------------------------------- 引数（元の main に --target などを足す）
def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--box", choices=["M"], required=True)
    r.add_argument("--arm", choices=list(C.ARMS), required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--tasks", type=int, default=50)
    r.add_argument("--n_per_task", type=int, required=True)
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--engine", choices=["graph", "eager"], default="graph")
    r.add_argument("--within_tasks", type=int, nargs="*", default=None)
    r.add_argument("--fresh_tasks", type=int, nargs="*", default=None)
    r.add_argument("--no_measure", action="store_true")
    r.add_argument("--ckpt", type=int, default=1)
    r.add_argument("--target", choices=["sample", "soft"], default="sample", help="soft = 腕 S2（真の分布を目標に）")
    r.add_argument("--pi_dir", default=None, help="先に計算した π の置き場（<pi_dir>/a<α>/seed<k>/t<t>.npz）")
    r.add_argument("--no_recorder", action="store_true", help="検査用: (r1)(r2)(r5) の記録器を外す")
    r.add_argument("--checks_mode", choices=["stop", "record"], default="stop")
    r.add_argument("--entropy_check", choices=["stop", "record"], default="stop", help="S-entropy だけを記録にする（主の決定が要る）")
    p = sub.add_parser("replay")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--n_per_task", type=int, default=10000)
    p.add_argument("--src", required=True, help="A2 の seed の dir（online.npz と ckpts/）")
    p.add_argument("--out", required=True)
    p.add_argument("--pi_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--engine", choices=["graph", "eager"], default="graph")
    a = ap.parse_args()
    if a.cmd == "run":
        C.run(a)
    else:
        replay(a)


if __name__ == "__main__":
    main()

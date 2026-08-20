"""teachw_0820 Phase 1: 教師複雑度スイープの本走 [spec_teachw_0820 §3, §5]。

  OMP_NUM_THREADS=1 .venv/bin/python -m src.teachw --config configs/teachw_0820.yaml
  OMP_NUM_THREADS=1 .venv/bin/python -m src.teachw --hidden 100     # 1 アームだけ
  OMP_NUM_THREADS=1 .venv/bin/python -m src.teachw --smoke          # 0->50k, results/_smoke_teachw

アーム = condA 教師 LTU 幅 H_T ∈ {1,2,4,8,32,100}。**入力ストリームは全アームで同一**で
なければ設計が成立しない (§1)。これは `train.make_gens` が入力 (`gens["input"]`) と
教師 (`gens["teacher"]`) を別 generator に分けており、かつ generator の種が
(exp, width) だけで決まる ―― H_T に依存しない ―― ことによる。H_T が変えるのは
`gens["teacher"]` からの抽選数だけで、`SCREnv` の乱数消費は 1 回も動かない。
実効検証は S2 (flip_state 軌跡 hash の全アーム一致) [§7]。

教師出力スケーリング y_scaled = y_raw·√(100/H_T) は `envs.LTUTarget.out_scale`。
H_T=100 では係数 1.0 = 厳密な恒等なので、このアームは ratchet_log_0819 と bit 一致する
(Phase 0-1 / S3 のアンカー)。

probe は ratchet_log と同じ読み取り専用フックだが、記録するのは §3 が要求する
厳密 p̂ と eval_loss_exact だけ (F 系の分解は本実験の判定に不要)。**probe の無擾乱性**は
`full_support_ro` を使うこと (本家 `SCREnv.full_support()` は `maybe_flip()` を呼んで
env.t を進める学習ループ用の関数) と、Phase 0-4 の bit 一致検査で担保する。
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import yaml

from .common import (ROOT, load_config, pick_device, build_runs, group_runs,
                     resolve_outdir)
from .ratchet_log import full_support_ro, teacher_f64, state_hash
from .train import train_group

# seed ごとに保存する時系列 (probe グリッド上)
UNIT_SERIES = ["p_hat"]                       # [n_rec, h]
RUN_SERIES = ["eval_loss_exact", "alive", "p_hat_median_alive", "var_y", "mean_y"]
# 最終記録点だけ保存する台帳用のユニット量 [h]
FINAL_UNIT = ["p_hat", "w_norm", "b", "v", "cos_u_mu"]


def out_scale_for(hidden, ref):
    """§3 のスケーリング則 y_scaled = y_raw·√(ref/H_T)。ref=H_T なら厳密に 1.0。"""
    return math.sqrt(float(ref) / float(hidden))


def arm_cfg(cfg, hidden):
    """アーム用に condA.target_hidden / target_out_scale を差し替えた config。"""
    c = copy.deepcopy(cfg)
    c["condA"]["target_hidden"] = int(hidden)
    c["condA"]["target_out_scale"] = out_scale_for(hidden, c["teachw"]["scale_ref"])
    return c


def arm_dir(outdir, hidden):
    return os.path.join(outdir, f"H{int(hidden)}")


# ---------------------------------------------------------------- 読み取り専用の厳密計測

def exact_record(st, tau):
    """1 記録点ぶんの 32 パターン厳密量 [§3]。全て float64 で計算し float32 で返す。

    p̂_i = P(w_iᵀx + b_i > 0) を入力分布の**全サポート**上で厳密に取る (eval バッチの
    有限標本ではない)。alive = #{i: p̂_i ≥ tau} が本実験の主判定量。
    ゲートの不等号 (>0) は `nets.VecMLP._gate` / `lop_metrics.open_frac` と同一。"""
    env, net, teacher = st["env"], st["net"], st["teacher"]
    if bool(st["centered"].any()):
        # centered だと x_in が running_mean に依存し µ̂ の解析形も変わる。黙って通さない。
        raise NotImplementedError("teachw は enc=std 専用 (§3)")
    with torch.no_grad():
        X = full_support_ro(env).double()                     # [P,R,m]
        y = teacher_f64(teacher, X)                           # [P,R]
        W, b = net.W.double(), net.b.double()
        v, c = net.v.double(), net.c.double()

        pre = torch.einsum("rhd,prd->prh", W, X) + b          # [P,R,h]
        gate = (pre > 0).double()
        a = torch.relu(pre)
        delta = (a * v).sum(dim=-1) + c - y                   # [P,R]

        p_hat = gate.mean(dim=0)                              # [R,h]
        alive = (p_hat >= tau)                                # [R,h]
        n_alive = alive.sum(dim=1)                            # [R]
        # alive ユニットの median p̂ (P3: surv_hist T1 の複雑度依存)。alive が無ければ NaN
        pm = torch.where(alive, p_hat, torch.full_like(p_hat, float("nan")))
        p_med = torch.nanquantile(pm, 0.5, dim=1, interpolation="linear")

        mu = X.mean(dim=0)                                    # [R,m] = flip ‖ 0.5·1
        w_norm = W.norm(dim=2)                                # [R,h]
        cos_u_mu = (torch.einsum("rhd,rd->rh", W, mu)
                    / (w_norm * mu.norm(dim=1)[:, None]).clamp_min(1e-300))

        cv = lambda t: t.detach().cpu().numpy().astype(np.float32)
        return dict(p_hat=cv(p_hat), w_norm=cv(w_norm), b=cv(b), v=cv(v),
                    cos_u_mu=cv(cos_u_mu),
                    eval_loss_exact=cv((delta ** 2).mean(dim=0)),
                    alive=cv(n_alive.double()), p_hat_median_alive=cv(p_med),
                    # Var[y_scaled] / E[y_scaled] は 32 パターン一様分布上の母分散・母平均。
                    # S4 (Var[y] 帯の本走ログ上での確認) [§7] をログだけで閉じるため記録する。
                    var_y=cv(y.var(dim=0, unbiased=False)), mean_y=cv(y.mean(dim=0)),
                    flip_state=cv(env.flip_state.double()))


class Recorder:
    """probe(st, step) として train_group に渡す読み取り専用アキュムレータ。

    `st` を読むだけで generator を消費せず env.t も進めない (`full_support_ro` を使う)。
    無擾乱性は Phase 0-4 で probe あり/なしの state hash 一致として実測する。"""

    def __init__(self, steps, R, h, f, tau):
        self.steps = np.asarray(sorted(steps), dtype=np.int64)
        self.index = {int(s): i for i, s in enumerate(self.steps)}
        n = len(self.steps)
        self.tau = float(tau)
        self.buf = {k: np.zeros((n, R, h), dtype=np.float32) for k in UNIT_SERIES}
        for k in RUN_SERIES:
            self.buf[k] = np.zeros((n, R), dtype=np.float32)
        self.buf["flip_state"] = np.zeros((n, R, f), dtype=np.float32)
        self.final = None
        self.filled = np.zeros(n, dtype=bool)
        self.n_calls = 0

    def __call__(self, st, step):
        i = self.index.get(int(step))
        if i is None:
            return
        rec = exact_record(st, self.tau)
        for k, arr in self.buf.items():
            arr[i] = rec[k]
        self.filled[i] = True
        self.n_calls += 1
        if i == len(self.steps) - 1:
            self.final = {k: rec[k] for k in FINAL_UNIT}

    def check_complete(self):
        miss = np.flatnonzero(~self.filled)
        if len(miss):
            raise RuntimeError(f"未記録の grid 点が {len(miss)} 個: {self.steps[miss][:10]} ...")


def record_steps(total, every):
    return sorted({0, total} | set(range(0, total + 1, int(every))))


def flip_hash(rec, i):
    """seed i の flip_state 軌跡の sha256 (§7 S2)。記録グリッドも hash に含める。"""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rec.steps).tobytes())
    h.update(np.ascontiguousarray(rec.buf["flip_state"][:, i]).tobytes())
    return h.hexdigest()


def write_logs(rec, runs, outdir, hidden, out_scale):
    """seed ごとに logs/seed{k}.npz [§8]。判定に使う生ログなので .gitignore しない。"""
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    paths = []
    for i, r in enumerate(runs):
        out = dict(step=rec.steps, run_id=np.array(r["run_id"]), seed=np.int64(r["seed"]),
                   target_hidden=np.int64(hidden), out_scale=np.float64(out_scale),
                   lr=np.float32(r["lr"]), period=np.int64(r["period"]),
                   width=np.int64(r["width"]), p_hat_tau=np.float32(rec.tau))
        for k, arr in rec.buf.items():
            out[k] = arr[:, i]
        for k, arr in (rec.final or {}).items():
            out[f"final_{k}"] = arr[i]
        p = os.path.join(logdir, f"seed{r['seed']}.npz")
        np.savez_compressed(p, **out)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- アーム実行

def run_arm(cfg, hidden, device, outdir, total_steps=None, snapshot=True):
    """1 アーム (教師幅 H_T) を最後まで走らせて logs / meta.json を書く。"""
    c = arm_cfg(cfg, hidden)
    C, P = c["common"], c["teachw"]
    total = int(total_steps or C["total_steps"])
    scale = c["condA"]["target_out_scale"]

    runs = build_runs(c)
    for r in runs:                                   # 来歴 (アーム軸は run_id に出ない)
        r["target_hidden"], r["out_scale"] = int(hidden), scale
    groups = group_runs(runs)
    if len(groups) != 1:
        raise ValueError(f"teachw は単一グループ前提 (§3) だが {len(groups)} 個: {sorted(groups)}")
    gkey, gruns = next(iter(groups.items()))

    adir = arm_dir(outdir, hidden)
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(c, fh, allow_unicode=True)
    with open(os.path.join(adir, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)

    steps = record_steps(total, P["probe_every"])
    R, h, f = len(gruns), int(gkey[1]), int(c["condA"]["f"])
    rec = Recorder(steps, R, h, f, P["p_hat_tau"])
    print(f"[H={hidden}] group={gkey} R={R} h={h} total={total} scale={scale:.6g} "
          f"記録点={len(steps)}", flush=True)

    t0 = time.time()
    st, elapsed = train_group(gkey, gruns, c, device, adir, total_steps=total,
                              probe=rec, probe_steps=steps,
                              snapshot_steps=[total] if snapshot else [])
    rec.check_complete()
    paths = write_logs(rec, gruns, adir, hidden, scale)

    alive_final = rec.buf["alive"][-1]
    meta = dict(target_hidden=int(hidden), out_scale=scale,
                elapsed_sec=round(time.time() - t0, 1), train_sec=round(elapsed, 1),
                device=device, date=time.strftime("%Y-%m-%d %H:%M:%S"),
                group=str(gkey), R=R, width=h, total_steps=total,
                period=int(gruns[0]["period"]), n_record_steps=len(steps),
                p_hat_tau=float(P["p_hat_tau"]),
                alive_final={int(r["seed"]): float(alive_final[i])
                             for i, r in enumerate(gruns)},
                flip_hash={int(r["seed"]): flip_hash(rec, i) for i, r in enumerate(gruns)},
                state_hash_final=state_hash(st),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                torch_num_threads=torch.get_num_threads(),
                logs_mb=round(sum(os.path.getsize(p) for p in paths) / 1e6, 2),
                spec="specs/spec_teachw_0820.md")
    with open(os.path.join(adir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)
    print(f"[H={hidden}] {elapsed:.1f}s ({total/elapsed:.0f} steps/s) "
          f"alive_final={alive_final.astype(int).tolist()} logs={meta['logs_mb']}MB", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/teachw_0820.yaml")
    ap.add_argument("--hidden", nargs="*", type=int, default=None,
                    help="実行するアーム (既定は teachw.hidden_values 全部)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="0->50k / seed 2 本 (results/_smoke_teachw)")
    ap.add_argument("--no-snapshot", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg["common"].update(total_steps=50000, seeds=[0, 1])
    if args.seeds is not None:
        cfg["common"]["seeds"] = list(args.seeds)
    if args.total_steps is not None:
        cfg["common"]["total_steps"] = int(args.total_steps)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)

    outdir = args.outdir or (os.path.join(ROOT, "results", "_smoke_teachw")
                             if args.smoke else resolve_outdir(args.config))
    os.makedirs(outdir, exist_ok=True)
    print(f"outdir: {outdir}", flush=True)
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)

    hiddens = args.hidden if args.hidden else list(cfg["teachw"]["hidden_values"])
    for hd in hiddens:
        run_arm(cfg, hd, device, outdir,
                total_steps=cfg["common"]["total_steps"], snapshot=not args.no_snapshot)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

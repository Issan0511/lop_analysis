"""bias 専用 weight decay 実験の共通機構 (パイロットと本走で共有)。

ここには**判定を一切置かない**。置くのは
  * 壁座標 (beta, kappa) の独立厳密計算と、凍結済み `exact_layer_record_p1` との突き合わせ
  * task 末ごとの記述統計レコーダ
  * `mlp2_phase1` の腕実行経路に `wd_b` を差し込む薄い runner
  * provenance / markdown の小道具
だけである。事前登録された判定規則は `src/bias_wd_0901.py` 側に置く。

壁の代数 [HANDOFF §2.1]。condA・32 パターン厳密サポートで、層の前活性
``z = W x_in + b`` はサポート上で ``s +- (W xi)`` の形を取る (``xi`` は自由座標の
中心化ゆらぎ)。したがって

    p_hat_i = 0  <=>  max_p z_{p,i} <= 0  <=>  beta_i + kappa_i <= 0
    beta_i  := mean_p z / sd_p z            (= exact_layer_record_p1 の M + B)
    kappa_i := (max_p z - mean_p z) / sd_p z

第1層では入力の自由座標が 0/1 の 5 ビットなので ``xi`` は ±1/2 で、
``kappa_i = ||w_free||_1 / ||w_free||_2 in [1, sqrt5]`` という閉形式に一致する。
上の定義は閉形式を仮定せずに全層へそのまま延長できるので、こちらを実装の主とし、
第1層で閉形式との一致を毎記録点で検査する (S3)。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import ROOT
from .mlp2_phase0 import identity_sanity_pass
from .mlp2_phase1 import (
    NUMERIC_DIVERGENCE,
    NumericDivergenceError,
    _arm,
    _base_cfg,
    _numeric_divergence_event,
    exact_layer_record_p1,
    setup_arm_p1,
    train_arm_p1,
)
from .ratchet_log import full_support_ro, teacher_f64


THIN_MAX = 8.0 / 32.0        # 「痩せた」発火率のしきい [HANDOFF §4]
SAT_MIN = 30.0 / 32.0        # 「常時発火」のしきい [HANDOFF §5.3 W4]


# ---------------------------------------------------------------- 小道具

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_omp(expected: int) -> dict:
    """`OMP_NUM_THREADS` の実測値を返す。不一致は即エラー。"""
    want, got = str(int(expected)), os.environ.get("OMP_NUM_THREADS")
    if got != want:
        raise RuntimeError(f"OMP_NUM_THREADS must be {want}, got {got!r}")
    return {"pass_": True, "expected": want, "actual": got}


def markdown_table(frame: pd.DataFrame) -> str:
    """依存を増やさない Markdown レンダラ (centered_freeze_0901 と同形)。"""
    columns = [str(column) for column in frame.columns]

    def render(value) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "n/a"
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def lam_tag(value: float) -> str:
    """lambda -> 腕名に使える短縮タグ。0 -> ``none``、1e-3 -> ``1em3``。"""
    value = float(value)
    if value == 0.0:
        return "none"
    exponent = math.log10(value)
    if abs(exponent - round(exponent)) > 1e-12:
        raise ValueError(f"lambda {value!r} is not a power of ten")
    exponent = int(round(exponent))
    return f"1e{exponent}" if exponent >= 0 else f"1em{-exponent}"


def _q(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _frac(mask: np.ndarray, valid: np.ndarray) -> float:
    n = int(valid.sum())
    return float((mask & valid).sum() / n) if n else float("nan")


# ---------------------------------------------------- 壁座標の独立厳密計算

def exact_wall_record(st: dict, sigma_tol: float) -> tuple[list[dict], dict]:
    """32 パターン厳密サポート上の (beta, kappa, sigma, p_hat, b) を層ごとに返す。

    `exact_layer_record_p1` とは**独立に**前向き計算をやり直す。両者の p_hat /
    sigma / beta / unfit が一致することが S3 の中身であり、片方のバグが
    もう片方に伝播しないようにするためにコードを共有しない。
    """
    net = st["net"]
    flags = st.get("centered_layers") or [False] * len(net.Ws)
    means = st.get("layer_means") or [None] * len(net.Ws)
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()             # [P,R,m]
        y = teacher_f64(st["teacher"], X)                   # [P,R]
        cur = X
        layers = []
        for li, (W0, b0) in enumerate(zip(net.Ws, net.bs)):
            W, b = W0.double(), b0.double()
            if flags[li]:
                cur = cur - means[li].double()[None]
            z = torch.einsum("rhd,prd->prh", W, cur) + b     # [P,R,h]
            pre_mean = z.mean(dim=0)
            pre_max = z.amax(dim=0)
            pre_sd = z.var(dim=0, unbiased=False).clamp_min(0).sqrt()
            p_hat = (z > 0).double().mean(dim=0)
            valid = pre_sd >= float(sigma_tol)
            beta = torch.full_like(pre_sd, float("nan"))
            kappa = torch.full_like(pre_sd, float("nan"))
            beta[valid] = pre_mean[valid] / pre_sd[valid]
            kappa[valid] = (pre_max[valid] - pre_mean[valid]) / pre_sd[valid]
            layers.append(dict(b=b, p_hat=p_hat, sigma=pre_sd, beta=beta,
                               kappa=kappa, pre_max=pre_max, pre_mean=pre_mean,
                               valid=valid, W=W))
            cur = torch.relu(z)
        yhat = (cur * net.v.double()).sum(dim=-1) + net.c.double()
        residual = yhat - y
        signal_var = y.var(dim=0, unbiased=False)
        run = dict(signal_var=signal_var,
                   residual_var=residual.var(dim=0, unbiased=False),
                   unfit=residual.var(dim=0, unbiased=False) / signal_var,
                   eval_loss_exact=residual.square().mean(dim=0))
    return layers, run


def wall_closed_form_kappa(W: torch.Tensor, n_flip: int) -> torch.Tensor:
    """第1層の閉形式 ``kappa = ||w_free||_1 / ||w_free||_2``。

    condA の入力は ``[flip_state (f), rnd (m-f)]`` の順で、32 パターンで動くのは
    末尾 ``m-f`` 座標だけ (`envs.SCREnv.step`)。
    """
    free = W[:, :, int(n_flip):]
    l1 = free.abs().sum(dim=2)
    l2 = free.norm(dim=2)
    out = torch.full_like(l2, float("nan"))
    ok = l2 > 0
    out[ok] = l1[ok] / l2[ok]
    return out


# ------------------------------------------------------------- レコーダ

RUN_KEYS = ("unfit", "eval_loss_exact", "signal_var", "residual_var")


class TaskEndRecorder:
    """task 末で厳密統計を、細かい格子で非有限ガードだけを回すレコーダ。

    * ``guard_steps``: 非有限検出 (S4)。`_numeric_divergence_event` は isfinite の
      走査だけなので細かく回してよい
    * ``record_steps``: 32 パターン厳密列挙。task 末のみ
    ここは記述統計だけを持つ。窓の切り出しも判定も呼び出し側の仕事。
    """

    def __init__(self, arm: str, wd_b: float, st: dict, *, record_steps,
                 guard_steps, guard_every: int, sigma_tol: float,
                 identity_tol: float, keep_unit_arrays: bool = True):
        self.arm = str(arm)
        self.wd_b = float(wd_b)
        self.record_steps = {int(v) for v in record_steps}
        self.guard_steps = {int(v) for v in guard_steps}
        self.guard_every = int(guard_every)
        self.sigma_tol = float(sigma_tol)
        self.identity_tol = float(identity_tol)
        self.keep_unit_arrays = bool(keep_unit_arrays)
        self.n_flip = int(st["env"].f)
        self.depth = len(st["net"].Ws)
        self.rows: list[dict] = []
        self.unit: dict[int, dict[str, np.ndarray]] = {}
        self.seen: set[int] = set()
        self.max_err = dict(mean=0.0, sd=0.0, wall=0.0, cos_mu=0.0,
                            beta=0.0, sigma=0.0, unfit=0.0, kappa_closed=0.0,
                            beta_elementwise_rel=0.0, beta_scale=0.0)
        self.n_quantization_violations = 0
        self.n_wall_identity_violations = 0
        self.n_nonfinite_required = 0
        self.n_records = 0

    # -- 1 記録点
    def __call__(self, st: dict, step: int) -> None:
        step = int(step)
        if step in self.guard_steps:
            event = _numeric_divergence_event(st, step)
            if event is not None:
                event["probe_every"] = self.guard_every
                raise NumericDivergenceError(event)
        if step not in self.record_steps:
            return
        if step in self.seen:
            raise RuntimeError(f"duplicate probe at step {step}")
        self.seen.add(step)

        ours, run = exact_wall_record(st, self.sigma_tol)
        reference, sanity = exact_layer_record_p1(st, self.sigma_tol)
        ref_layers, ref_run = reference["layers"], reference["run"]
        self._cross_check(ours, run, ref_layers, ref_run, sanity)

        for ri, meta in enumerate(st["runs"]):
            row = {"arm": self.arm, "wd_b": self.wd_b, "step": step,
                   "task": step // int(meta["period"]), "seed": int(meta["seed"])}
            for key in RUN_KEYS:
                row[key] = float(run[key][ri].item())
            for li, layer in enumerate(ours, start=1):
                row.update(self._layer_stats(layer, ref_layers[li - 1], li, ri))
            self.rows.append(row)

        if self.keep_unit_arrays:
            self.unit[step] = {
                f"layer{li}_{key}": layer[key].detach().cpu().numpy().astype(np.float32)
                for li, layer in enumerate(ours, start=1)
                for key in ("b", "p_hat", "sigma", "beta", "kappa")
            }
        self.n_records += 1

    # -- 2 実装の突き合わせ (S3)
    def _cross_check(self, ours, run, ref_layers, ref_run, sanity) -> None:
        for li, (a, b) in enumerate(zip(ours, ref_layers), start=1):
            valid = a["valid"] & torch.isfinite(b["denom"])
            if valid.any():
                self.max_err["sigma"] = max(self.max_err["sigma"], _relerr(
                    a["sigma"][valid], b["denom"][valid]))
                # beta は符号を変えながら 0 を通過する量なので、要素ごとの
                # 相対誤差は分母ゼロで発散する。実装一致の判定には
                # 「量のスケールで割った誤差」を使い、要素ごとの相対誤差は
                # 診断として併記だけする [bias_wd_0901 S3 追補]。
                beta_theirs = (b["M"] + b["B"])[valid]
                self.max_err["beta"] = max(self.max_err["beta"], _scaled_err(
                    a["beta"][valid], beta_theirs))
                self.max_err["beta_elementwise_rel"] = max(
                    self.max_err["beta_elementwise_rel"],
                    _relerr(a["beta"][valid], beta_theirs))
                self.max_err["beta_scale"] = max(
                    self.max_err["beta_scale"],
                    float(beta_theirs.abs().max().item()))
            if not torch.equal(a["p_hat"], b["p_hat"]):
                self.max_err["wall"] = float("inf")
            # p_hat は 1/32 格子
            grid = a["p_hat"] * 32.0
            self.n_quantization_violations += int(
                (grid - grid.round()).abs().gt(0).sum().item())
            # 壁恒等式: p_hat == 0  <=>  beta + kappa <= 0
            gap = a["beta"] + a["kappa"]
            dead = a["p_hat"] == 0
            disagree = (dead != (gap <= 0)) & a["valid"] & gap.abs().gt(1e-9)
            self.n_wall_identity_violations += int(disagree.sum().item())
            if li == 1:
                closed = wall_closed_form_kappa(a["W"], self.n_flip)
                ok = a["valid"] & torch.isfinite(closed)
                if ok.any():
                    self.max_err["kappa_closed"] = max(
                        self.max_err["kappa_closed"],
                        _relerr(a["kappa"][ok], closed[ok]))
            s = sanity["layers"][li - 1]
            self.max_err["mean"] = max(self.max_err["mean"], s["mean_max_relerr"])
            self.max_err["sd"] = max(self.max_err["sd"], s["sd_max_relerr"])
            self.max_err["cos_mu"] = max(self.max_err["cos_mu"],
                                         s["l1_cos_mu_max_relerr"])
            if not s["finite_required"]:
                self.n_nonfinite_required += 1
        self.max_err["unfit"] = max(self.max_err["unfit"],
                                    _relerr(run["unfit"], ref_run["unfit"]))
        if not sanity["run_finite"]:
            self.n_nonfinite_required += 1

    # -- 3 1 層 1 seed の記述統計
    def _layer_stats(self, ours: dict, theirs: dict, li: int, ri: int) -> dict:
        p = ours["p_hat"][ri].cpu().numpy()
        b = ours["b"][ri].cpu().numpy()
        beta = ours["beta"][ri].cpu().numpy()
        kappa = ours["kappa"][ri].cpu().numpy()
        sigma = ours["sigma"][ri].cpu().numpy()
        valid = ours["valid"][ri].cpu().numpy() & np.isfinite(beta) & np.isfinite(kappa)
        alive = (p > 0) & valid
        wall_frac = np.abs(beta) / kappa
        margin = kappa * sigma - np.abs(b)
        pre = f"L{li}_"
        return {
            pre + "strict_dead_frac": float((p == 0).mean()),
            pre + "alive": int(alive.sum()),
            pre + "n_invalid": int((~valid).sum()),
            pre + "b_median_alive": _q(b[alive], 0.5),
            pre + "b_q25_alive": _q(b[alive], 0.25),
            pre + "b_q75_alive": _q(b[alive], 0.75),
            pre + "b_median_all": _q(b, 0.5),
            pre + "b_maxabs": float(np.abs(b).max()),
            pre + "wall_frac": _q(wall_frac[alive], 0.5),
            # B = b/sigma。HANDOFF §2.5 の「alive 中央 b 項」はこの量である
            # (committed logs で median_B が -0.75 -> -0.91 / +0.17 -> +12.85 /
            #  -0.75 -> -1.18 を再現することを確認済み)。
            pre + "B_median_alive": _q((b / np.where(sigma > 0, sigma, np.nan))[alive], 0.5),
            pre + "beta_median_alive": _q(beta[alive], 0.5),
            pre + "kappa_median_alive": _q(kappa[alive], 0.5),
            pre + "sigma_median_alive": _q(sigma[alive], 0.5),
            pre + "margin_median_alive": _q(margin[alive], 0.5),
            pre + "p_hat_median_alive": _q(p[alive], 0.5),
            pre + "p_hat_thin_frac": _frac(p <= THIN_MAX, alive),
            pre + "p_hat_sat_frac": _frac(p >= SAT_MIN, alive),
            pre + "eff_rank": float(theirs["eff_rank"][ri].item()),
            pre + "eff_rank_W": float(theirs["eff_rank_W"][ri].item()),
            pre + "w_norm_median": float(theirs["w_norm_median"][ri].item()),
            pre + "wcos_mean": float(theirs["wcos_mean"][ri].item()),
        }

    # -- 4 まとめ
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def sanity(self) -> dict:
        missing = sorted(self.record_steps - self.seen)
        identity_ok = all(self.max_err[k] <= self.identity_tol
                          for k in ("mean", "sd", "cos_mu", "beta", "sigma",
                                    "unfit", "kappa_closed"))
        return dict(
            arm=self.arm, wd_b=self.wd_b,
            n_record_steps=len(self.record_steps), n_recorded=len(self.seen),
            missing_steps=missing[:10], n_missing=len(missing),
            max_relerr={k: float(v) for k, v in self.max_err.items()},
            identity_tol=self.identity_tol,
            n_quantization_violations=int(self.n_quantization_violations),
            n_wall_identity_violations=int(self.n_wall_identity_violations),
            n_nonfinite_required=int(self.n_nonfinite_required),
            pass_=bool(not missing and identity_ok
                       and self.n_quantization_violations == 0
                       and self.n_wall_identity_violations == 0
                       and self.n_nonfinite_required == 0),
        )


def _scaled_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a-b| / max|b|。0 を通過する量の実装一致に使う正しい尺度。

    要素ごとの相対誤差は分母が 0 に近づくと発散するので、beta のように符号を
    変えながら 0 を横切る量には使えない。ベクトルとしてのスケールで割る。
    """
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    scale = float(b.abs().max().item())
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    err = (a - b).abs()
    err = err[torch.isfinite(err)]
    return float(err.max().item()) / scale if err.numel() else 0.0


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a-b| / max(|b|, tiny)。両方 float64 前提。"""
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    denom = b.abs().clamp_min(1e-300)
    err = ((a - b).abs() / denom)
    err = err[torch.isfinite(err)]
    return float(err.max().item()) if err.numel() else 0.0


# ------------------------------------------------------------- 腕の実行

def run_arm(cfg: dict, arm_name: str, wd_b: float, outdir: Path, *,
            total_steps: int, task_period: int, guard_every: int,
            device: str = "cpu", keep_unit_arrays: bool = True,
            write_logs: bool = True) -> dict:
    """1 腕 (= 1 lambda 水準) を seeds 全部まとめて走らせる。

    腕の設定は凍結済みの `setup_arm_p1` にそのまま任せ、`wd_b` だけを構築後に
    差し込む。`set_weight_decay_b` は乱数も状態も消費しないので、`wd_b=0` の腕は
    既存 `mlp2_phase1_0829` の対応腕と bit 一致する (S0/S1 で実測確認する)。
    """
    base = _base_cfg(cfg)
    arm_cfg = _arm(cfg, arm_name)
    P = cfg["phase1"]
    st = setup_arm_p1(base, arm_cfg, device)
    st["net"].set_weight_decay_b(wd_b)

    _, before = exact_layer_record_p1(st, float(P["sigma_degenerate_tol"]))
    if not identity_sanity_pass(before, float(cfg["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{arm_name}: preflight identity failed")

    record_steps = list(range(0, total_steps + 1, task_period))
    guard_steps = list(range(0, total_steps + 1, guard_every))
    probe_steps = sorted(set(record_steps) | set(guard_steps))
    rec = TaskEndRecorder(arm_name, wd_b, st, record_steps=record_steps,
                          guard_steps=guard_steps, guard_every=int(guard_every),
                          sigma_tol=float(P["sigma_degenerate_tol"]),
                          identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
                          keep_unit_arrays=keep_unit_arrays)
    checkpoints = [int(v) for v in cfg["common"].get("checkpoints", [])
                   if int(v) <= total_steps]
    print(f"[{arm_name}] wd_b={wd_b:g} hidden={arm_cfg['hidden']} "
          f"centered={arm_cfg['centered_layers']} steps={total_steps:,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_p1(st, rec, probe_steps, total_steps, outdir,
                               checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(registered_total_steps=int(total_steps),
                     elapsed_sec=float(elapsed),
                     detection="nonfinite_training_state_at_probe",
                     action="mark_arm_failed_and_continue",
                     exclude_partial_logs_from_analysis=True, rescue="none")
        path = outdir / "arm_status" / f"{arm_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"[{arm_name}] {NUMERIC_DIVERGENCE} at step "
              f"{event['detected_step']:,} seeds={event['bad_seeds']}", flush=True)
        return dict(arm=arm_name, wd_b=wd_b, status=NUMERIC_DIVERGENCE,
                    elapsed_sec=elapsed, divergence=event,
                    sanity=rec.sanity(), frame=rec.dataframe())

    sanity = rec.sanity()
    frame = rec.dataframe()
    if write_logs:
        write_arm_npz(outdir, arm_name, wd_b, st, rec)
    print(f"[{arm_name}] complete in {elapsed:.1f}s "
          f"(sanity {'PASS' if sanity['pass_'] else 'FAIL'})", flush=True)
    return dict(arm=arm_name, wd_b=wd_b, status="COMPLETE", elapsed_sec=elapsed,
                sanity=sanity, frame=frame,
                final_state=dict(b_maxabs=float(max(
                    b.abs().max().item() for b in st["net"].bs))))


def write_arm_npz(outdir: Path, arm: str, wd_b: float, st: dict,
                  rec: TaskEndRecorder) -> list[Path]:
    """seed ごとの per-unit 時系列 (gitignore 対象の生ログ)。"""
    logdir = Path(outdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(sorted(rec.unit), dtype=np.int64)
    if not steps.size:
        return []
    keys = sorted(rec.unit[int(steps[0])])
    stacked = {k: np.stack([rec.unit[int(s)][k] for s in steps]) for k in keys}
    paths = []
    for ri, meta in enumerate(st["runs"]):
        payload = dict(step=steps, arm=np.array(arm), wd_b=np.float64(wd_b),
                       seed=np.int64(meta["seed"]),
                       task_period=np.int64(meta["period"]))
        payload.update({k: v[:, ri] for k, v in stacked.items()})
        path = logdir / f"{arm}_seed{int(meta['seed'])}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


# ------------------------------------------------------------- provenance

def provenance(experiment: str, cfg_path: Path, cfg: dict, outdir: Path,
               extra: dict, started: float, command: list[str],
               output_names) -> dict:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                        text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = None, []
    outputs = {}
    for name in output_names:
        path = Path(outdir) / name
        if path.exists():
            outputs[name] = sha_file(path)
    spec = cfg.get("spec")
    spec_path = Path(ROOT) / spec if spec else None
    return {
        "experiment": experiment,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "command": command,
        "elapsed_sec": round(time.time() - started, 3),
        "cwd": os.getcwd(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "device": "cpu",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "git_hash": git_hash, "git_dirty": dirty,
        "config": str(cfg_path), "config_sha256": sha_file(cfg_path),
        "spec": spec,
        "spec_sha256": sha_file(spec_path) if spec_path and spec_path.exists() else None,
        "output_sha256": outputs,
        **extra,
    }

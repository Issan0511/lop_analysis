"""本走 `bias_wd_0901`: condA・centered で bias だけに weight decay を掛ける。

事前登録: `specs/spec_bias_wd_0901.md`（この実装より**先に**単独 commit されている）。
lambda グリッドは `results/bias_wd_pilot_0901/grid_selection.json` が凍結済み規則で
決めた値を config へ写したもの。ここでは判定規則だけを実装する。

コマンド::

    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_0901 --s0
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_0901 --s1s2
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_0901 --smoke
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_0901 --arm W1_main
    OMP_NUM_THREADS=1 .venv/bin/python -m src.bias_wd_0901 --analyze-only
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .bias_wd_common import (
    markdown_table, provenance, require_omp, run_arm,
)
from .common import ROOT, load_config
from .dose_const_5m import clopper_pearson
from .mlp2_phase0 import make_gens
from .mlp2_phase0b import _ci_components
from .nets import VecMLPL

CONFIG = Path(ROOT) / "configs" / "bias_wd_0901.yaml"

# 判定名（spec §5.2 / §5.3）
BIAS_WD_PROTECTS = "BIAS_WD_PROTECTS"
DEAD_ONLY = "DEAD_ONLY"
PAYS_STATIC_COST = "PAYS_STATIC_COST"
LEVEL_ONLY_NO_KINETICS = "LEVEL_ONLY_NO_KINETICS"
NO_EFFECT = "NO_EFFECT"


# --------------------------------------------------------------- config

def arms_of(cfg: dict) -> list[dict]:
    return list(cfg["arms"])


def arm_lambda(cfg: dict, name: str) -> float:
    return float(next(a for a in cfg["arms"] if a["name"] == name)["wd_b"])


def depth_of(cfg: dict, name: str) -> int:
    return len(next(a for a in cfg["arms"] if a["name"] == name)["hidden"])


def floor_for(cfg: dict, name: str) -> float:
    key = "unfit_floor_L1" if depth_of(cfg, name) == 1 else "unfit_floor_L2"
    return float(cfg["bias_wd"][key])


def validate_config(cfg: dict, *, full: bool) -> None:
    C, A, P = cfg["common"], cfg["condA"], cfg["bias_wd"]
    if cfg["common"].get("device", "cpu") != "cpu":
        raise ValueError("bias_wd_0901 is CPU-only")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"])) != (20, 15, 100):
        raise ValueError("registered condA dimensions differ")
    if float(A["beta"]) != 0.7 or list(A["T_values"]) != [10_000]:
        raise ValueError("registered condA regime differs")
    if float(cfg["intervention"]["center_alpha"]) != 0.01:
        raise ValueError("registered center_alpha differs")
    if float(C["lr_main"]) != 0.01:
        raise ValueError("registered lr differs")
    if (int(P["bootstrap_B"]), int(P["bootstrap_seed"])) != (20_000, 20_260_902):
        raise ValueError("registered bootstrap differs")
    if list(P["late_block_tasks"]) != [451, 500] or list(P["early_block_tasks"]) != [51, 100]:
        raise ValueError("registered windows differ")
    if float(P["dead_threshold"]) != 0.232 or float(P["equivalence_margin"]) != 0.10:
        raise ValueError("registered thresholds differ")
    if (float(P["unfit_floor_L1"]), float(P["unfit_floor_L2"])) != (1e-16, 1e-23):
        raise ValueError("registered floors differ")
    names = [a["name"] for a in cfg["arms"]]
    required = ["W1_none", "W1_main", "W1_sub1", "W1_sub2", "W1_sub3",
                "W2_Aall_none", "W2_Aall_main", "W2_Aall_sub"]
    if names != required:
        raise ValueError(f"registered arms differ: {names}")
    if arm_lambda(cfg, "W1_none") != 0.0 or arm_lambda(cfg, "W2_Aall_none") != 0.0:
        raise ValueError("control arms must have wd_b=0")
    if arm_lambda(cfg, "W1_main") != arm_lambda(cfg, "W2_Aall_main"):
        raise ValueError("the two main arms must share the main lambda")
    if full and (int(C["total_steps"]) != 5_000_000
                 or list(C["seeds"]) != list(range(10))):
        raise ValueError("the full run is 5M steps and seeds 0..9")


def outdir_of(cfg: dict) -> Path:
    return Path(ROOT) / cfg["bias_wd"]["output_dir"]


# ------------------------------------------------------------ S0 / S1 / S2

def s0_replay(cfg: dict, outdir: Path) -> dict:
    """wd_b=0 の 2 腕が committed `mlp2_phase1_0829` の対応腕と一致すること。

    深さ 1 は `L1w100_A1`、深さ 2 は `L2_Aall` と突き合わせる。WD コード経路を
    通したうえでの一致なので、S0 が通れば S1（wd_b=0 で bit 一致）も同時に立つ。
    """
    steps = int(cfg["bias_wd"]["s0_replay_steps"])
    base_dir = Path(ROOT) / cfg["baseline_dir"] / "logs"
    pairs = {"W1_none": cfg["bias_wd"]["baseline_arm_L1"],
             "W2_Aall_none": cfg["bias_wd"]["baseline_arm_L2"]}
    replay_cfg = copy.deepcopy(cfg)
    replay_cfg["common"]["total_steps"] = steps
    replay_cfg["common"]["checkpoints"] = []

    report, ok = {}, True
    for arm, baseline in pairs.items():
        result = run_arm(replay_cfg, arm, 0.0, outdir, total_steps=steps,
                         task_period=1000, guard_every=1000,
                         keep_unit_arrays=False, write_logs=False)
        frame = result["frame"]
        diffs, max_abs = [], {"unfit": 0.0, "eval_loss_exact": 0.0}
        depth = depth_of(cfg, arm)
        for seed in cfg["common"]["seeds"]:
            path = base_dir / f"{baseline}_seed{int(seed)}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            mine = frame[frame.seed == int(seed)].set_index("step")
            with np.load(path, allow_pickle=False) as z:
                for step in mine.index:
                    idx = np.flatnonzero(z["step"] == int(step))
                    if len(idx) != 1:
                        diffs.append(dict(seed=int(seed), step=int(step),
                                          field="step", detail=str(len(idx))))
                        continue
                    i = int(idx[0])
                    for key in ("unfit", "eval_loss_exact"):
                        delta = abs(float(mine.loc[step, key]) - float(z[key][i]))
                        max_abs[key] = max(max_abs[key], delta)
                        if delta > 1e-12:
                            diffs.append(dict(seed=int(seed), step=int(step),
                                              field=key, detail=f"abs={delta:.3g}"))
                    for li in range(1, depth + 1):
                        p = z[f"layer{li}_p_hat"][i]
                        dead = float((p == 0).mean())
                        if float(mine.loc[step, f"L{li}_strict_dead_frac"]) != dead:
                            diffs.append(dict(seed=int(seed), step=int(step),
                                              field=f"L{li}_strict_dead_frac",
                                              detail="mismatch"))
        report[arm] = dict(pass_=not diffs, baseline=baseline, steps=steps,
                           n_seeds=len(cfg["common"]["seeds"]),
                           n_probes=int(frame.step.nunique()), max_abs=max_abs,
                           differences=diffs[:50],
                           recorder_sanity=result["sanity"])
        ok = ok and not diffs
        print(f"S0 [{arm} vs {baseline}]: {'PASS' if not diffs else 'FAIL'} "
              f"max|dunfit|={max_abs['unfit']:.3g}", flush=True)

    out = dict(pass_=bool(ok), arms=report)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "s0_replay.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if not ok:
        raise RuntimeError("S0 replay failed; the full run is blocked")
    return out


def s1_s2_algebra(cfg: dict, outdir: Path) -> dict:
    """S1/S2: WD が触るのは隠れ層 bias だけであることの代数的確認。

    同一状態から 1 step 進め、``wd_b=0`` と ``wd_b=lambda`` の差分を見る。
      * ``Ws``・``v``・``c`` は bit 一致でなければならない（S2）
      * ``bs[i]`` の差は厳密に ``-lr * lambda * b_before`` でなければならない
      * ``wd_b=0`` の経路は WD を含まない参照実装と bit 一致（S1）
    """
    lam = float(cfg["bias_wd"]["s2_probe_lambda"])
    device, m = "cpu", int(cfg["condA"]["m"])
    rows = []
    for hidden in ([100], [100, 100]):
        gens = make_gens("A", hidden[0], device)
        ref = VecMLPL(4, hidden, m, gens["init"], device)
        state = ref.state_dict()
        lr = torch.full((4,), float(cfg["common"]["lr_main"]))
        g = torch.Generator().manual_seed(4242)
        for b in ref.bs:                       # b=0 初期化だと WD が恒等になる
            b.copy_(torch.randn(b.shape, generator=g))
        state = ref.state_dict()
        gWs = [torch.randn(4, h, i, generator=g) for h, i in
               zip(hidden, [m] + hidden[:-1])]
        gbs = [torch.randn(4, h, generator=g) for h in hidden]
        gv = torch.randn(4, hidden[-1], generator=g)
        gc = torch.randn(4, generator=g)

        def stepped(wd_b, plain=False):
            net = VecMLPL(4, hidden, m, make_gens("A", hidden[0], device)["init"],
                          device)
            net.load_state(state)
            if plain:                          # WD を含まない参照実装
                for i in range(net.L):
                    net.Ws[i] -= lr[:, None, None] * gWs[i]
                    net.bs[i] -= lr[:, None] * gbs[i]
                net.v -= lr[:, None] * gv
                net.c -= lr * gc
            else:
                net.set_weight_decay_b(wd_b)
                net.sgd_step_layers(lr, gWs, gbs, gv, gc)
            return net

        zero, plain, decayed = stepped(0.0), stepped(0.0, plain=True), stepped(lam)
        s1 = all(torch.equal(a, b) for a, b in
                 zip(zero.Ws + zero.bs + [zero.v, zero.c],
                     plain.Ws + plain.bs + [plain.v, plain.c]))
        s2_untouched = (all(torch.equal(a, b) for a, b in zip(zero.Ws, decayed.Ws))
                        and torch.equal(zero.v, decayed.v)
                        and torch.equal(zero.c, decayed.c))
        # b の差は -lr*lambda*b_before。float32 の 2 回の減算の丸めが残るので、
        # 厳密 0 ではなく b のスケールの数 ULP を許容にする。
        eps = float(torch.finfo(zero.bs[0].dtype).eps)
        err, tol, signal = 0.0, 0.0, 0.0
        for i in range(len(hidden)):
            before = state[f"b{i + 1}" if len(hidden) > 1 else "b"]
            expected = -lr[:, None] * lam * before
            err = max(err, float((decayed.bs[i] - zero.bs[i] - expected).abs().max()))
            tol = max(tol, 4.0 * eps * float(before.abs().max()))
            signal = max(signal, float(expected.abs().max()))
        rows.append(dict(depth=len(hidden), lam=lam, S1_bit_identity=bool(s1),
                         S2_W_v_c_untouched=bool(s2_untouched),
                         S2_bias_delta_max_abs_err=err,
                         S2_bias_delta_tol_ulp=tol,
                         S2_bias_delta_signal=signal,
                         S2_bias_delta_ok=bool(err <= tol)))

    # ソース側の確認: sgd_step_layers の中で wd_b を参照する更新は
    # self.bs[i] の 1 本だけであること（docstring は AST では拾われない）。
    tree = ast.parse((Path(ROOT) / "src" / "nets.py").read_text(encoding="utf-8"))
    updates = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "sgd_step_layers", "sgd_step"):
            for aug in [n for n in ast.walk(node) if isinstance(n, ast.AugAssign)]:
                names = {n.id for n in ast.walk(aug.value) if isinstance(n, ast.Name)}
                attrs = {n.attr for n in ast.walk(aug.value)
                         if isinstance(n, ast.Attribute)}
                updates.setdefault(node.name, []).append(
                    dict(target=ast.unparse(aug.target),
                         uses_wd_b=bool("wd_b" in names or "wd_b" in attrs)))
    s2_source = all(
        [u["target"] for u in us if u["uses_wd_b"]] in (["self.bs[i]"], ["self.b"])
        for us in updates.values()) and len(updates) == 2

    out = dict(rows=rows, s2_source_only_bias_update_uses_wd_b=bool(s2_source),
               s2_source_updates=updates,
               pass_=bool(all(r["S1_bit_identity"] and r["S2_W_v_c_untouched"]
                              and r["S2_bias_delta_ok"] for r in rows) and s2_source))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "s1_s2_algebra.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"S1/S2: {'PASS' if out['pass_'] else 'FAIL'}", flush=True)
    if not out["pass_"]:
        raise RuntimeError(f"S1/S2 failed: {out}")
    return out


# ---------------------------------------------------------------- blocks

METRIC_COLUMNS = [
    "unfit", "eval_loss_exact",
    "L1_strict_dead_frac", "L1_b_median_alive", "L1_wall_frac",
    "L1_beta_median_alive", "L1_kappa_median_alive", "L1_sigma_median_alive",
    "L1_margin_median_alive", "L1_p_hat_median_alive", "L1_p_hat_thin_frac",
    "L1_p_hat_sat_frac", "L1_eff_rank", "L1_eff_rank_W", "L1_w_norm_median",
    "L1_wcos_mean",
]
L2_COLUMNS = [c.replace("L1_", "L2_") for c in METRIC_COLUMNS if c.startswith("L1_")]


def block_levels(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """50 task 刻みのブロック平均。unfit は床を当ててから log10 を取る。"""
    size = int(cfg["bias_wd"]["block_tasks"])
    rows = []
    for (arm, seed), g in frame.groupby(["arm", "seed"], sort=True):
        floor = floor_for(cfg, arm)
        g = g[g.task > 0].copy()
        g["block"] = ((g["task"] - 1) // size + 1).astype(int)
        clipped = np.maximum(g["unfit"].to_numpy(), floor)
        g["log10_unfit"] = np.log10(clipped)
        g["at_floor"] = g["unfit"].to_numpy() <= floor
        present = [c for c in METRIC_COLUMNS + L2_COLUMNS if c in g.columns]
        for block, gb in g.groupby("block"):
            row = dict(arm=arm, seed=int(seed), block=int(block),
                       task_lo=int(gb.task.min()), task_hi=int(gb.task.max()),
                       n_task_ends=int(len(gb)),
                       mean_log10_unfit=float(gb["log10_unfit"].mean()),
                       log10_mean_unfit=float(np.log10(
                           np.maximum(gb["unfit"].mean(), floor))),
                       floor=floor, floor_frac=float(gb["at_floor"].mean()))
            row.update({c: float(gb[c].mean()) for c in present})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "seed", "block"])


def _draws(cfg: dict, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(cfg["bias_wd"]["bootstrap_seed"]))
    return rng.integers(0, n, size=(int(cfg["bias_wd"]["bootstrap_B"]), n))


def paired_ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    """percentile を主、studentized を退化検出つきで併記する（spec §5.5）。"""
    P = cfg["bias_wd"]
    out = _ci_components(np.asarray(values, dtype=np.float64), draws, "mean",
                         float(P["degenerate_se_tol"]),
                         float(P["degenerate_frac_max"]),
                         float(P["degenerate_width_ratio_max"]))
    out["ci_lo"], out["ci_hi"] = out["percentile_ci_lo"], out["percentile_ci_hi"]
    out["ci_basis"] = "percentile"
    return out


def _seed_series(levels: pd.DataFrame, arm: str, block: int, column: str,
                 seeds: list[int]) -> np.ndarray:
    g = levels[(levels.arm == arm) & (levels.block == block)].set_index("seed")
    missing = [s for s in seeds if s not in g.index]
    if missing:
        raise RuntimeError(f"{arm} block {block}: missing seeds {missing}")
    return g.loc[seeds, column].to_numpy(dtype=np.float64)


# -------------------------------------------------------------- verdicts

def classify_main(a: bool, b: bool, c: bool) -> tuple[str, bool]:
    """spec §5.2 の決定木。返り値は (判定名, DEAD_ONLY フラグ)。"""
    if not a:
        return NO_EFFECT, False
    if not b:
        return PAYS_STATIC_COST, (not c)
    if not c:
        return LEVEL_ONLY_NO_KINETICS, False
    return BIAS_WD_PROTECTS, False


def analyze(cfg: dict, outdir: Path) -> dict:
    P = cfg["bias_wd"]
    seeds = [int(s) for s in cfg["common"]["seeds"]]
    frame = pd.read_csv(outdir / "task_end_metrics.csv")
    levels = block_levels(cfg, frame)
    levels.to_csv(outdir / "block_levels.csv", index=False)

    size = int(P["block_tasks"])
    b02 = (int(P["early_block_tasks"][1]) - 1) // size + 1
    b10 = (int(P["late_block_tasks"][1]) - 1) // size + 1
    draws = _draws(cfg, len(seeds))
    rows, details = [], {}

    def series(arm, block, column):
        return _seed_series(levels, arm, block, column, seeds)

    # ---- 主判定（W1_main のみ。副 3 水準は REPORT_ONLY）
    main, ctrl = "W1_main", "W1_none"
    dead_main = series(main, b10, "L1_strict_dead_frac")
    dead_ctrl = series(ctrl, b10, "L1_strict_dead_frac")
    dead_ci = paired_ci(cfg, dead_main - dead_ctrl, draws)
    cond_a = bool(float(dead_main.mean()) <= float(P["dead_threshold"]))

    u_main10, u_ctrl10 = (series(main, b10, "mean_log10_unfit"),
                          series(ctrl, b10, "mean_log10_unfit"))
    level_ci = paired_ci(cfg, u_main10 - u_ctrl10, draws)
    margin = float(P["equivalence_margin"])
    cond_b = bool(level_ci["ci_hi"] < margin)

    u_main02, u_ctrl02 = (series(main, b02, "mean_log10_unfit"),
                          series(ctrl, b02, "mean_log10_unfit"))
    drift = (u_main10 - u_main02) - (u_ctrl10 - u_ctrl02)
    drift_ci = paired_ci(cfg, drift, draws)
    cond_c = bool(drift_ci["ci_hi"] < 0.0)

    verdict, dead_only = classify_main(cond_a, cond_b, cond_c)
    n_below = int((dead_main <= float(P["dead_threshold"])).sum())
    cp_lo, cp_hi = clopper_pearson(n_below, len(seeds))
    rows.append(dict(
        pred="P-main", scope=f"tasks {P['late_block_tasks'][0]}-{P['late_block_tasks'][1]} "
        f"(block {b10}) W1_main lambda={arm_lambda(cfg, main):g}",
        verdict=verdict, dead_only_flag=int(dead_only),
        cond_a=int(cond_a), cond_b=int(cond_b), cond_c=int(cond_c),
        evidence=(
            f"(a) dead {float(dead_ctrl.mean()):.6f} -> {float(dead_main.mean()):.6f} "
            f"(threshold {float(P['dead_threshold']):.3f}; ratio "
            f"{float(dead_main.mean()) / max(float(dead_ctrl.mean()), 1e-300):.4f}; "
            f"paired {dead_ci['point']:+.6f} CI [{dead_ci['ci_lo']:+.6f}, "
            f"{dead_ci['ci_hi']:+.6f}]); "
            f"(b) mean(log10 unfit) {float(u_ctrl10.mean()):+.4f} -> "
            f"{float(u_main10.mean()):+.4f}, paired {level_ci['point']:+.4f} CI "
            f"[{level_ci['ci_lo']:+.4f}, {level_ci['ci_hi']:+.4f}] vs margin "
            f"{margin:+.2f}; (c) B10-B02 drift diff {drift_ci['point']:+.4f} CI "
            f"[{drift_ci['ci_lo']:+.4f}, {drift_ci['ci_hi']:+.4f}]"),
        n_seeds_below_dead_threshold=n_below, cp95_lo=cp_lo, cp95_hi=cp_hi,
        ci_basis="percentile",
        ci_degenerate=int(dead_ci["ci_degenerate"] or level_ci["ci_degenerate"]
                          or drift_ci["ci_degenerate"])))
    details["main"] = dict(dead=dead_ci, level=level_ci, drift=drift_ci,
                           cond=dict(a=cond_a, b=cond_b, c=cond_c))

    # ---- 副 3 水準は用量反応の記述のみ
    for arm in ("W1_sub1", "W1_sub2", "W1_sub3"):
        d = series(arm, b10, "L1_strict_dead_frac")
        u = series(arm, b10, "mean_log10_unfit")
        dr = (u - series(arm, b02, "mean_log10_unfit")) - (u_ctrl10 - u_ctrl02)
        rows.append(dict(
            pred="P-dose", scope=f"block {b10} {arm} lambda={arm_lambda(cfg, arm):g}",
            verdict="REPORT_ONLY", dead_only_flag=0, cond_a="", cond_b="", cond_c="",
            evidence=(f"dead {float(d.mean()):.6f}; mean(log10 unfit) "
                      f"{float(u.mean()):+.4f}; B10-B02 drift diff vs none "
                      f"{float(dr.mean()):+.4f}"),
            n_seeds_below_dead_threshold="", cp95_lo="", cp95_hi="",
            ci_basis="", ci_degenerate=""))

    # ---- W2: 上方暴走（飽和・ランク崩壊）の抑制
    w2_main, w2_ctrl = "W2_Aall_main", "W2_Aall_none"
    er_main10 = series(w2_main, b10, "L1_eff_rank")
    er_ctrl10 = series(w2_ctrl, b10, "L1_eff_rank")
    er_ctrl02 = series(w2_ctrl, b02, "L1_eff_rank")
    er_ci = paired_ci(cfg, er_main10 - er_ctrl10, draws)
    keep = float(P["eff_rank_keep_frac"])
    w2_ok = bool(er_ci["ci_lo"] > 0.0
                 and float(np.median(er_main10)) >= keep * float(np.median(er_ctrl02)))
    rows.append(dict(
        pred="W2", scope=f"block {b10} layer-1 activation eff_rank ({w2_main})",
        verdict="SATURATION_PREVENTED" if w2_ok else "NOT_PREVENTED",
        dead_only_flag=0, cond_a="", cond_b="", cond_c="",
        evidence=(f"none B02 {float(np.median(er_ctrl02)):.4f} -> none B10 "
                  f"{float(np.median(er_ctrl10)):.4f}; lambda arm B10 "
                  f"{float(np.median(er_main10)):.4f}; keep>= {keep:.2f}x B02 = "
                  f"{keep * float(np.median(er_ctrl02)):.4f}; paired "
                  f"{er_ci['point']:+.4f} CI [{er_ci['ci_lo']:+.4f}, "
                  f"{er_ci['ci_hi']:+.4f}]"),
        n_seeds_below_dead_threshold="", cp95_lo="", cp95_hi="",
        ci_basis="percentile", ci_degenerate=int(er_ci["ci_degenerate"])))
    details["W2"] = er_ci

    # ---- W3: 保護マージンの時間署名
    m02 = series(main, b02, "L1_margin_median_alive")
    m10 = series(main, b10, "L1_margin_median_alive")
    m_ci = paired_ci(cfg, m10 - m02, draws)
    w3 = ("MARGIN_WIDENS" if m_ci["ci_lo"] > 0 else
          "MARGIN_NARROWS" if m_ci["ci_hi"] < 0 else "FLAT")
    ctrl_m_ci = paired_ci(cfg, series(ctrl, b10, "L1_margin_median_alive")
                          - series(ctrl, b02, "L1_margin_median_alive"), draws)
    rows.append(dict(
        pred="W3", scope=f"{main} layer-1 alive median margin (kappa*sigma - |b|), B02->B10",
        verdict=w3, dead_only_flag=0, cond_a="", cond_b="", cond_c="",
        evidence=(f"{float(m02.mean()):+.5f} -> {float(m10.mean()):+.5f}; paired "
                  f"{m_ci['point']:+.5f} CI [{m_ci['ci_lo']:+.5f}, {m_ci['ci_hi']:+.5f}]"
                  f"; W1_none same contrast {ctrl_m_ci['point']:+.5f} CI "
                  f"[{ctrl_m_ci['ci_lo']:+.5f}, {ctrl_m_ci['ci_hi']:+.5f}]"),
        n_seeds_below_dead_threshold="", cp95_lo="", cp95_hi="",
        ci_basis="percentile", ci_degenerate=int(m_ci["ci_degenerate"])))
    details["W3"] = dict(main=m_ci, none=ctrl_m_ci)

    # ---- W4: 飽和側の記述
    for arm in (w2_ctrl, w2_main, "W2_Aall_sub"):
        sat = series(arm, b10, "L1_p_hat_sat_frac")
        sat02 = series(arm, b02, "L1_p_hat_sat_frac")
        bm = series(arm, b10, "L1_b_median_alive")
        rows.append(dict(
            pred="W4", scope=f"block {b10} layer-1 alive p_hat>=30/32 ({arm})",
            verdict="REPORT_ONLY", dead_only_flag=0, cond_a="", cond_b="", cond_c="",
            evidence=(f"sat_frac B02 {float(np.median(sat02)):.4f} -> B10 "
                      f"{float(np.median(sat)):.4f}; alive median b B10 "
                      f"{float(np.median(bm)):+.4f}"),
            n_seeds_below_dead_threshold="", cp95_lo="", cp95_hi="",
            ci_basis="", ci_degenerate=""))

    # ---- S5: 恒真ガード（最大 lambda が frozen へ漸近するか。記録のみ）
    strongest = min(((a["name"], float(a["wd_b"])) for a in cfg["arms"]
                     if a["name"].startswith("W1_") and float(a["wd_b"]) > 0),
                    key=lambda kv: -kv[1])[0]
    rows.append(dict(
        pred="S5", scope=f"tautology guard ({strongest}, strongest W1 lambda)",
        verdict="REPORT_ONLY", dead_only_flag=0, cond_a="", cond_b="", cond_c="",
        evidence=(f"alive median b B10 "
                  f"{float(np.median(series(strongest, b10, 'L1_b_median_alive'))):+.5f}; "
                  f"dead {float(series(strongest, b10, 'L1_strict_dead_frac').mean()):.6f}; "
                  f"unfit(mean over seeds of block mean) "
                  f"{float(np.mean(10 ** series(strongest, b10, 'mean_log10_unfit'))):.6g}"),
        n_seeds_below_dead_threshold="", cp95_lo="", cp95_hi="",
        ci_basis="", ci_degenerate=""))

    verdict_frame = pd.DataFrame(rows)
    verdict_frame.to_csv(outdir / "verdict.csv", index=False)

    endpoints = pd.DataFrame({"seed": seeds})
    for arm in [a["name"] for a in cfg["arms"]]:
        for block, tag in ((b02, "B02"), (b10, "B10")):
            endpoints[f"{arm}_{tag}_dead"] = series(arm, block, "L1_strict_dead_frac")
            endpoints[f"{arm}_{tag}_meanlog10unfit"] = series(arm, block, "mean_log10_unfit")
            endpoints[f"{arm}_{tag}_b_median_alive"] = series(arm, block, "L1_b_median_alive")
            endpoints[f"{arm}_{tag}_eff_rank_L1"] = series(arm, block, "L1_eff_rank")
    endpoints.to_csv(outdir / "paired_endpoints.csv", index=False)

    _figure(cfg, frame, levels, outdir)
    result = dict(main_verdict=verdict, dead_only=dead_only,
                  conditions=dict(a=cond_a, b=cond_b, c=cond_c),
                  W2="SATURATION_PREVENTED" if w2_ok else "NOT_PREVENTED",
                  W3=w3, blocks=dict(B02=b02, B10=b10), details=details)
    _summary(cfg, outdir, verdict_frame, levels, result)
    return result


# --------------------------------------------------------------- outputs

def _figure(cfg: dict, frame: pd.DataFrame, levels: pd.DataFrame,
            outdir: Path) -> None:
    w1 = [a["name"] for a in cfg["arms"] if a["name"].startswith("W1_")]
    w2 = [a["name"] for a in cfg["arms"] if a["name"].startswith("W2_")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(w1)))
    panels = [("L1_strict_dead_frac", "strict_dead_frac (L1)", False),
              ("L1_b_median_alive", "alive median b (L1)", False),
              ("unfit", "exact-support unfit", True)]
    for (key, label, logy), ax in zip(panels, axes[0]):
        for color, arm in zip(colors, w1):
            g = frame[frame.arm == arm].groupby("task")[key].median()
            ax.plot(g.index, g.values, lw=1.2, color=color,
                    label=f"$\\lambda$={arm_lambda(cfg, arm):g}")
        ax.set_xlabel("task")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[0, 0].axhline(float(cfg["bias_wd"]["dead_threshold"]), color="gray",
                       ls="--", lw=1)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("W1 (depth 1, width 100)")

    colors2 = plt.cm.plasma(np.linspace(0, 0.75, len(w2)))
    panels2 = [("L1_eff_rank", "layer-1 activation eff_rank", False),
               ("L1_p_hat_sat_frac", "alive p_hat >= 30/32 (L1)", False),
               ("unfit", "exact-support unfit", True)]
    for (key, label, logy), ax in zip(panels2, axes[1]):
        for color, arm in zip(colors2, w2):
            g = frame[frame.arm == arm].groupby("task")[key].median()
            ax.plot(g.index, g.values, lw=1.2, color=color,
                    label=f"$\\lambda$={arm_lambda(cfg, arm):g}")
        ax.set_xlabel("task")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("W2_Aall (depth 2, both layers centered)")
    fig.suptitle("bias_wd_0901 — condA centered, weight decay on hidden bias only")
    fig.tight_layout()
    fig.savefig(outdir / "fig_bias_wd.png", dpi=150)
    plt.close(fig)


def _summary(cfg: dict, outdir: Path, verdict: pd.DataFrame,
             levels: pd.DataFrame, result: dict) -> None:
    P = cfg["bias_wd"]
    b02, b10 = result["blocks"]["B02"], result["blocks"]["B10"]
    late = levels[levels.block == b10]
    table = (late.groupby("arm")[["L1_strict_dead_frac", "mean_log10_unfit",
                                  "log10_mean_unfit", "L1_b_median_alive",
                                  "L1_wall_frac", "L1_margin_median_alive",
                                  "L1_eff_rank", "L1_p_hat_sat_frac",
                                  "floor_frac"]]
             .mean().reindex([a["name"] for a in cfg["arms"]]).reset_index())
    table.insert(1, "wd_b", [arm_lambda(cfg, a) for a in table["arm"]])
    lines = [
        "# bias_wd_0901 — 本走の結果", "",
        f"事前登録: [`{cfg['spec']}`](../../{cfg['spec']})。"
        f"lambda グリッドは `results/bias_wd_pilot_0901/grid_selection.json` が"
        f"凍結済み規則で決めた値。", "",
        "## Verdict", "", markdown_table(verdict.drop(columns=["cond_a", "cond_b", "cond_c"])), "",
        f"## 終盤窓（task {P['late_block_tasks'][0]}–{P['late_block_tasks'][1]}"
        f" = block {b10}）の腕別水準（seed 平均）", "",
        markdown_table(table), "",
        "## 集計の約束", "",
        f"- 主判定に使うのは **`mean(log10 unfit)`**（seed 内でブロック内 task 末の "
        f"log10 を平均）。`log10(mean unfit)` も上表に併記するが判定には使わない",
        f"- 床は系ごとに別。深さ1系 `{float(P['unfit_floor_L1']):g}`"
        f"（`dose_const_5m_0830` の S6 較正を継承）、深さ2系 "
        f"`{float(P['unfit_floor_L2']):g}`（`mlp2_phase1_0829` の S6 較正を継承）。"
        f"本走では再較正しない",
        f"- ブロックは {int(P['block_tasks'])} task 刻み。B02 = task "
        f"{P['early_block_tasks'][0]}–{P['early_block_tasks'][1]}、B10 = task "
        f"{P['late_block_tasks'][0]}–{P['late_block_tasks'][1]}",
        f"- CI は seed 水準の paired percentile bootstrap（B={int(P['bootstrap_B'])}、"
        f"seed {int(P['bootstrap_seed'])}）。studentized も計算して "
        f"`ci_degenerate` を出すが、**主は percentile**（この repo では Phase 0b 以降"
        f"ほぼ全行で studentized が退化する）",
        "- 二値割合の CI は Clopper–Pearson", "",
        "## 引いてはいけない線（HANDOFF §7）", "",
        "1. 高 $\\lambda$ 端で dead が 0 になることを証拠にしない。$b\\equiv0$ かつ "
        "centered なら task 末に消灯できないのは恒等式の帰結であって観測ではない",
        "2. 「WD が LoP を治す」と一般に書かない。スコープは condA・centered・幅100・"
        "$T=10^4$・batch=1・lr=0.01・5M に限る",
        "3. std 腕へ外挿しない（台帳の逃げ道が std にはある）",
        "4. 新規性は「WD が効く」ではなく「**$b$ だけの減衰で足りるか**」に置く。"
        "先行（`docs/lit_bias_wd_0901.md`）では全パラメータ L2 が dead を上げ "
        "effective rank を下げており、符号が逆である",
        "5. `strict_dead` の低下を機能改善と読み替えない。(a) と (b)(c) は独立に読む",
        "6. パイロット（`results/bias_wd_pilot_0901/`）の数値を結果として引用しない",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- driver

OUTPUTS = ("verdict.csv", "summary.md", "paired_endpoints.csv",
           "task_end_metrics.csv", "block_levels.csv", "run_sanity.json",
           "config_used.yaml", "fig_bias_wd.png")


def _shard(outdir: Path) -> Path:
    path = outdir / "shards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--s0", action="store_true")
    parser.add_argument("--s1s2", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    validate_config(cfg, full=not args.smoke)
    require_omp(int(cfg["bias_wd"]["omp_num_threads"]))
    outdir = Path(args.outdir).resolve() if args.outdir else outdir_of(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    gate_dir = Path(ROOT) / "results" / "_gate_bias_wd_0901"
    started = time.time()

    if args.s1s2:
        s1_s2_algebra(cfg, gate_dir)
        return
    if args.s0:
        s0_replay(cfg, gate_dir)
        return
    if args.smoke:
        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg["common"]["seeds"] = [0]
        smoke_dir = Path(ROOT) / "results" / "_smoke_bias_wd_0901"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for arm in cfg["arms"]:
            r = run_arm(smoke_cfg, arm["name"], float(arm["wd_b"]), smoke_dir,
                        total_steps=30_000, task_period=10_000, guard_every=1_000,
                        keep_unit_arrays=False, write_logs=False)
            if not r["sanity"]["pass_"]:
                raise RuntimeError(f"smoke sanity failed: {r['sanity']}")
        print("SMOKE PASS", flush=True)
        return

    for name, path in (("S1/S2", gate_dir / "s1_s2_algebra.json"),
                       ("S0", gate_dir / "s0_replay.json")):
        if not path.exists():
            raise FileNotFoundError(f"run --s1s2 and --s0 before the full run ({name})")
        if not json.loads(path.read_text(encoding="utf-8")).get("pass_"):
            raise RuntimeError(f"saved {name} gate did not pass")

    total = int(cfg["common"]["total_steps"])
    period = int(cfg["phase1"]["task_period"])
    guard = int(cfg["bias_wd"]["guard_every"])
    todo = ([a for a in cfg["arms"] if a["name"] == args.arm] if args.arm
            else list(cfg["arms"]))
    if args.arm and not todo:
        raise SystemExit(f"unknown arm {args.arm!r}")

    if not args.analyze_only:
        for arm in todo:
            result = run_arm(cfg, arm["name"], float(arm["wd_b"]), outdir,
                             total_steps=total, task_period=period,
                             guard_every=guard)
            result["frame"].to_csv(_shard(outdir) / f"{arm['name']}.csv", index=False)
            (_shard(outdir) / f"{arm['name']}.json").write_text(json.dumps(
                {k: v for k, v in result.items() if k != "frame"},
                indent=2, ensure_ascii=False), encoding="utf-8")
            if result["status"] != "COMPLETE":
                print(f"[{arm['name']}] {result['status']}: continuing", flush=True)
            elif not result["sanity"]["pass_"]:
                raise RuntimeError(f"{arm['name']} sanity failed")
        if args.arm:
            return

    shards, meta = [], {}
    for arm in cfg["arms"]:
        path = _shard(outdir) / f"{arm['name']}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        shards.append(pd.read_csv(path))
        meta[arm["name"]] = json.loads(
            (_shard(outdir) / f"{arm['name']}.json").read_text(encoding="utf-8"))
    frame = pd.concat(shards, ignore_index=True)
    frame.to_csv(outdir / "task_end_metrics.csv", index=False)

    s0 = json.loads((gate_dir / "s0_replay.json").read_text(encoding="utf-8"))
    s1s2 = json.loads((gate_dir / "s1_s2_algebra.json").read_text(encoding="utf-8"))
    run_sanity = {
        "S0_wd0_arms_match_mlp2_phase1_0829": bool(s0["pass_"]),
        "S0_detail": {a: {"baseline": v["baseline"], "max_abs": v["max_abs"],
                          "n_probes": v["n_probes"]} for a, v in s0["arms"].items()},
        "S1_wd0_bit_identical_to_no_wd_path": bool(all(
            r["S1_bit_identity"] for r in s1s2["rows"])),
        "S2_W_v_c_untouched_by_wd": bool(all(
            r["S2_W_v_c_untouched"] and r["S2_bias_delta_max_abs_err"] == 0.0
            for r in s1s2["rows"])
            and s1s2["s2_source_only_bias_update_uses_wd_b"]),
        "S3_exact_support_identities": {
            a: dict(max_relerr=meta[a]["sanity"]["max_relerr"],
                    n_quantization_violations=meta[a]["sanity"]["n_quantization_violations"],
                    n_wall_identity_violations=meta[a]["sanity"]["n_wall_identity_violations"],
                    pass_=meta[a]["sanity"]["pass_"])
            for a in meta},
        "S3_pass": bool(all(meta[a]["sanity"]["pass_"] for a in meta)),
        "S4_numeric_divergence": {a: meta[a]["status"] for a in meta},
        "S4_probe_every": guard,
        "S5_tautology_guard": "reported in verdict.csv (REPORT_ONLY)",
        "training_elapsed_sec": {a: meta[a]["elapsed_sec"] for a in meta},
    }
    (outdir / "run_sanity.json").write_text(
        json.dumps(run_sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    with (outdir / "config_used.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    result = analyze(cfg, outdir)
    (outdir / "provenance.json").write_text(json.dumps(provenance(
        "bias_wd_0901", cfg_path, cfg, outdir,
        dict(analysis=result, run_sanity=run_sanity,
             pilot="results/bias_wd_pilot_0901/grid_selection.json"),
        started, sys.argv, OUTPUTS), indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(pd.read_csv(outdir / "verdict.csv")[
        ["pred", "scope", "verdict"]].to_string(index=False), flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)


if __name__ == "__main__":
    main()

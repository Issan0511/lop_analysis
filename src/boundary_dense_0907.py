# -*- coding: utf-8 -*-
"""boundary_dense_0907: タスク境界まわりを対数格子で記録し、沈下の収支を分ける。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.boundary_dense_0907 --arm BDref_1216
    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.boundary_dense_0907 --all

spec `specs/spec_boundary_dense_0907.md`。**学習は 1 bit も変えない**——変えるのは
(1) 記録格子（`probe_steps`）と (2) 新しい列 `layer1_b` だけ。`edge_law_0905` の runner・
recorder・書き出しは 1 行も書き換えず、`EdgeRecorder` を継承して列を足すだけにしてある。

probe が読み取り専用であること（`full_support_ro` は env にも生成器にも書かない）が
この実験の生命線で、それは仮定でなく **S-null（`LRoff0_1216` と state_hash が一致）** で
実測する。`ratchet_log` の同じ罠の注記も参照。
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from . import edge_law_0905 as E
from .common import ROOT, load_config

CONFIG = Path(ROOT) / "configs" / "boundary_dense_0907.yaml"
PERIOD = E.PERIOD


# ---------------------------------------------------------------------------
# 記録格子（spec §2-1）
# ---------------------------------------------------------------------------
def probe_steps(total: int, window, offsets, period: int = PERIOD) -> list[int]:
    """全タスク終端 ＋ 窓の中の各タスク開始からの対数格子。

    `ratchet_log.record_steps` と同じ流儀（境界のまわりだけ密・step 0 と total は必ず入る）。
    ただし窓は「タスク開始 t0 からの片側オフセット」で、反転**後**だけを見る。
    """
    steps = set(range(0, int(total) + 1, int(period)))
    lo, hi = int(window[0]), int(window[1])
    for t0 in range(lo, min(hi, int(total)) + 1, int(period)):
        for d in offsets:
            s = t0 + int(d)
            if s <= int(total):
                steps.add(s)
    steps.add(0)
    steps.add(int(total))
    return sorted(steps)


# ---------------------------------------------------------------------------
# 記録器（spec §2-2）
# ---------------------------------------------------------------------------
class BoundaryRecorder(E.EdgeRecorder):
    """``EdgeRecorder`` ＋ ユニット別バイアス列 ``b``（書き出しで ``layer1_b`` になる）。

    既存の列は 1 つも変えない。``b`` は学習状態を**読むだけ**で、乱数も env も触らない。
    """

    def __init__(self, steps: list[int], st: dict, *, record_units: bool = True):
        super().__init__(steps, st, record_units=record_units)
        if self.record_units:
            n, runs, width = len(self.steps), st["R"], st["hidden"][0]
            self.unit["b"] = np.empty((n, runs, width), dtype=np.float32)

    def __call__(self, st: dict, step: int) -> None:
        super().__call__(st, step)
        if not self.record_units:
            return
        i = self.index.get(int(step))
        if i is None:
            return
        self.unit["b"][i] = (st["net"].bs[0].detach().cpu().numpy()
                             .astype(np.float32))


# ---------------------------------------------------------------------------
# 腕の実行（`edge_law_0905._run_arm_edge` の写し・差は probes と recorder の 2 行）
# ---------------------------------------------------------------------------
def _run_arm_boundary(cfg: dict, arm: str, device: str, outdir: Path,
                      seeds: list[int], total: int, grid: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = seeds
    st = E.setup_arm_dial(c, E._arm(c, arm), device)
    E._apply_hook(st, _hook_of(arm), arm)
    every = int(c["common"]["lop_every"])
    probes = probe_steps(total, grid["window"], grid["offsets"])   # ← 差 1
    _, sanity0 = E.exact_layer_record_elu(st, E.SIGMA_TOL)
    if not E.identity_sanity_pass(sanity0, E.IDENTITY_TOL):
        raise E.SanityError(f"{arm} initial exact-support identity failed")
    train_arm_gate = E._train_fn(st)
    rec = BoundaryRecorder(probes, st)                             # ← 差 2
    checkpoints = [int(v) for v in c["common"].get("checkpoints", []) if int(v) <= total]
    print(f"[{arm}] act={st['activation']} dial={st['act_alpha']:g} "
          f"dose={st.get('target_dose')} seeds={seeds} steps={total:,} "
          f"probes={len(probes):,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_gate(st, rec, probes, total, outdir, checkpoints)
    except E.NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation"], act_alpha=st["act_alpha"],
                     family=st.get("family"), elapsed_sec=float(elapsed),
                     detection="nonfinite_training_state_at_probe",
                     partial_logs_excluded=True, rescue="none")
        path = E._arm_status_path(outdir, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{arm}] {E.NUMERIC_DIVERGENCE} at step {event['detected_step']:,}", flush=True)
        return dict(status=E.NUMERIC_DIVERGENCE, elapsed_sec=elapsed, divergence=event,
                    sanity=dict(pass_=False, numeric_divergence=True, event=event))
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise E.SanityError(f"{arm} exact-support sanity failed: {sanity}")
    E.write_arm_logs_edge(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity,
                n_probes=len(probes))


def registered() -> dict:
    return load_config(str(CONFIG))


def _hook_of(arm: str):
    for row in registered()["arms"]:
        if str(row["name"]) == str(arm):
            return row["hook"]
    raise KeyError(arm)


def grid_of(cfg: dict | None = None) -> dict:
    r = (cfg or registered())["record"]
    return dict(window=[int(v) for v in r["boundary_window"]],
                offsets=[int(v) for v in r["boundary_offsets"]])


def run_arm(arm: str, steps: int | None = None, outdir: Path | None = None,
            seeds: list[int] | None = None) -> dict:
    """1 腕を走らせて logs を置く（`edge_law_0905.run_single_arm` の写し）。"""
    reg = registered()
    row = {str(r["name"]): r for r in reg["arms"]}[str(arm)]
    cfg = E.build_cfg(CONFIG)
    E.require_omp(cfg)
    total = int(steps) if steps else int(row["total_steps"])
    cfg = copy.deepcopy(cfg)
    cfg["common"]["total_steps"] = total
    cfg["common"]["checkpoints"] = [int(v) for v in row["checkpoints"]]
    out = Path(outdir) if outdir is not None else (Path(ROOT) / reg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    use = [int(v) for v in (seeds if seeds is not None else cfg["common"]["seeds"])]
    started = time.time()
    grid = grid_of(reg)
    if steps:                       # 短縮走行では窓も総 step に合わせて畳む
        grid["window"] = [min(grid["window"][0], max(0, total - PERIOD)),
                          max(0, total - PERIOD)]
    result = _run_arm_boundary(cfg, arm, "cpu", out, use, total, grid)
    status = dict(result)
    status.update(arm=arm, total_steps=total, seeds=use,
                  wall_sec=time.time() - started, hook=row["hook"],
                  git_head=E._git_head(), grid=grid)
    p = E._arm_status_path(out, arm).with_name(f"{arm}_done.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    seeds = [int(v) for v in a.seeds.split(",")] if a.seeds else None
    arms = ([str(r["name"]) for r in registered()["arms"]] if a.all
            else [a.arm] if a.arm else [])
    if not arms:
        raise SystemExit("--arm NAME か --all を指定")
    for arm in arms:
        run_arm(arm, steps=a.steps,
                outdir=Path(a.outdir) if a.outdir else None, seeds=seeds)


if __name__ == "__main__":
    main()

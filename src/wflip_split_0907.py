# -*- coding: utf-8 -*-
"""wflip_split_0907: 蓄積した Δw_flip の向きを測るために ``layer1_w_flip`` を足す。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.wflip_split_0907 --arm WSref_1216
    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.wflip_split_0907 --all

spec `specs/spec_wflip_split_0907.md`（事前登録 `c9a2005`）。**学習は 1 bit も変えない**
——変えるのは記録列 `layer1_w_flip`（タスク終端のみ）だけ。記録格子・`layer1_b` は
`boundary_dense_0907` をそのまま継承し、`edge_law_0905` は 1 行も書き換えない。

書き出しは宿主 `E.write_arm_logs_edge` にそのまま書かせてから、新しい 2 列だけを
足して置き直す（既存列が宿主の関数から出ていることを構造的に保証するため）。
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from . import boundary_dense_0907 as BD
from . import edge_law_0905 as E
from .common import ROOT, load_config

CONFIG = Path(ROOT) / "configs" / "wflip_split_0907.yaml"
PERIOD = E.PERIOD
FLIP_SLICE = slice(0, 15)
N_FLIP = 15


def unit_w_flip_record(st: dict):
    """flip 15 ビットの重み ``W[:, :, 0:15]``（spec §3）。読むだけ。"""
    return st["net"].Ws[0][:, :, FLIP_SLICE]


# ---------------------------------------------------------------------------
# 記録器（`BoundaryRecorder` ＋ タスク終端の w_flip）
# ---------------------------------------------------------------------------
class WFlipRecorder(BD.BoundaryRecorder):
    """``BoundaryRecorder``（= 宿主 ＋ ``layer1_b``）に ``w_flip`` を足す。

    記録点は ``layer1_w_free`` と同じタスク終端の格子。学習状態を**読むだけ**で、
    乱数も env も触らない。
    """

    def __init__(self, steps: list[int], st: dict, *, record_units: bool = True):
        super().__init__(steps, st, record_units=record_units)
        task_end = (self.steps % PERIOD == 0)
        self.w_flip_steps = self.steps[task_end].astype(np.int64)
        self.w_flip_index = {int(v): i for i, v in enumerate(self.w_flip_steps)}
        if self.record_units:
            runs, width = st["R"], st["hidden"][0]
            self.w_flip = np.empty((len(self.w_flip_steps), runs, width, N_FLIP),
                                   dtype=np.float32)
        else:
            self.w_flip = np.empty((0, 0, 0, N_FLIP), dtype=np.float32)

    def __call__(self, st: dict, step: int) -> None:
        super().__call__(st, step)
        if not self.record_units:
            return
        j = self.w_flip_index.get(int(step))
        if j is None:
            return
        self.w_flip[j] = (unit_w_flip_record(st).detach().cpu().numpy()
                          .astype(np.float32))


def write_arm_logs_wflip(outdir: Path, arm: str, st: dict,
                         rec: WFlipRecorder) -> list[Path]:
    """宿主に書かせてから ``layer1_w_flip`` / ``layer1_w_flip_step`` を足す。

    既存列は 1 つも触らない（触れないように、宿主が書いた npz を読み直して
    そのまま書き戻す）。
    """
    paths = E.write_arm_logs_edge(outdir, arm, st, rec)
    for ri, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files}
        if "layer1_w_flip" in payload:
            raise RuntimeError("layer1_w_flip is already written by the host")
        payload["layer1_w_flip"] = rec.w_flip[:, ri]
        payload["layer1_w_flip_step"] = rec.w_flip_steps
        np.savez_compressed(path, **payload)
    return paths


# ---------------------------------------------------------------------------
# 腕の実行（`boundary_dense_0907._run_arm_boundary` の写し・差は recorder と書き出し）
# ---------------------------------------------------------------------------
def _run_arm_wflip(cfg: dict, arm: str, device: str, outdir: Path,
                   seeds: list[int], total: int, grid: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = seeds
    st = E.setup_arm_dial(c, E._arm(c, arm), device)
    E._apply_hook(st, _hook_of(arm), arm)
    every = int(c["common"]["lop_every"])
    probes = BD.probe_steps(total, grid["window"], grid["offsets"])
    _, sanity0 = E.exact_layer_record_elu(st, E.SIGMA_TOL)
    if not E.identity_sanity_pass(sanity0, E.IDENTITY_TOL):
        raise E.SanityError(f"{arm} initial exact-support identity failed")
    train_arm_gate = E._train_fn(st)
    rec = WFlipRecorder(probes, st)                                # ← 差 1
    checkpoints = [int(v) for v in c["common"].get("checkpoints", []) if int(v) <= total]
    print(f"[{arm}] act={st['activation']} dial={st['act_alpha']:g} "
          f"dose={st.get('target_dose')} seeds={seeds} steps={total:,} "
          f"probes={len(probes):,} w_flip_recs={len(rec.w_flip_steps):,}", flush=True)
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
    write_arm_logs_wflip(outdir, arm, st, rec)                     # ← 差 2
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity,
                n_probes=len(probes), n_w_flip=len(rec.w_flip_steps))


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
    result = _run_arm_wflip(cfg, arm, "cpu", out, use, total, grid)
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

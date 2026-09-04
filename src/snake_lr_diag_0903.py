"""snake_lr_diag_0903 — Snake を完走させる診断走（**未登録・事後**）。

    OMP_NUM_THREADS=1 python3 -m src.snake_lr_diag_0903 --stage run --lr 0.005
    OMP_NUM_THREADS=1 python3 -m src.snake_lr_diag_0903 --stage run --alpha 3
    OMP_NUM_THREADS=1 python3 -m src.snake_lr_diag_0903 --stage analyze --alpha 3

**2026-09-04 に振動数 alpha も振れるよう一般化した**（起票時は lr だけだった）。
登録値は (lr, alpha) = (0.01, 1.0)。**どちらか一方だけを動かす**のがこの診断の作法で、
両方同時に動かすと軸が 2 本になるので拒否する。

**これは `comb_isolate_0903` の登録腕ではない。** 登録された `SN_a1_1216` は lr 0.01 で
step 1,000 の最初の probe で seed 3 が非有限になり、spec §6 の登録どおり腕ごと落ちている
（`results/comb_isolate_0903/arm_status/SN_a1_1216.json`）。本モジュールはその後に
Issa の指示（2026-09-04）で回す**診断**で、変えた軸は **lr 1 本だけ**である。

**使ってよい読み方と、いけない読み方:**

* ○ 「Snake は lr を下げれば 5M 完走するか」「完走したとき LoP は出るか」
* ✗ 他の腕（`R_1216` / `LR_1216` / `CB1l` …）との**水準の比較**。lr が違うので
  同じハーネスではなく、5M step での学習の進み具合も違う
* ✗ 登録判定 V5 / V6 への算入。**verdict には一切入らない**

出力は `results/_diag_snake_lr_0903/lr<値>/` に分ける（登録走のディレクトリを汚さない）。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import ROOT, load_config
from .comb_isolate_0903 import CONFIG as STAGE_A_CONFIG
from .comb_isolate_0903 import _unit_tail
from .gate_dial_0902 import _load_new_arm
from .mlp2_phase0 import _sha_file, require_omp, write_csv
from .mlp2_phase1 import NUMERIC_DIVERGENCE
from .weird_act_0903 import (ONSET_CENSOR_AT, _onset_stats, _onset_times,
                             _run_arm_weird, _s_cap)

EXPERIMENT = "snake_lr_diag_0903"
ARM = "SN_a1_1216"
REGISTERED_LR = 0.01
REGISTERED_ALPHA = 1.0
def _unregistered_note(lr: float, alpha: float) -> str:
    axis = ("lr（0.01 → %g）" % lr if float(alpha) == REGISTERED_ALPHA
            else "dial = act_alpha（1.0 → %g）" % alpha)
    same_harness = float(lr) == REGISTERED_LR
    return ("この走は事前登録されていない。Issa の指示で回した診断で、変えた軸は "
            + axis + " の 1 本だけ。**verdict には入れない。** "
            + ("lr は登録値 0.01 なので他腕と同じハーネスに乗っており、水準の比較は"
               "意味を持つ（ただし事後・未登録）。"
               if same_harness else
               "lr が登録値と違うので他腕と同じハーネスではなく、**水準の比較には使わない**。"))


UNREGISTERED = _unregistered_note(0.005, 1.0)   # 起票時（lr 走）の文言


def _outdir(lr: float, alpha: float = REGISTERED_ALPHA) -> Path:
    root = Path(ROOT) / "results" / "_diag_snake_lr_0903"
    if float(alpha) == REGISTERED_ALPHA:          # 起票時の命名を保つ
        return root / ("lr" + ("%g" % lr).replace(".", "p"))
    return root / ("lr" + ("%g" % lr).replace(".", "p")
                   + "_a" + ("%g" % alpha).replace(".", "p"))


def _cfg_for(lr: float, alpha: float = REGISTERED_ALPHA) -> dict:
    """登録値から **1 軸だけ**動かした config を返す。"""
    same_lr = float(lr) == REGISTERED_LR
    same_alpha = float(alpha) == REGISTERED_ALPHA
    if same_lr and same_alpha:
        raise ValueError("登録値そのもの。診断は lr か alpha を変えたときだけ")
    if not same_lr and not same_alpha:
        raise ValueError("lr と alpha を同時に動かさない（軸が 2 本になる）")
    cfg = load_config(str(STAGE_A_CONFIG))
    cfg["common"]["lr_main"] = float(lr)
    for arm in cfg["arms"]:
        if arm["name"] == ARM:
            arm["dial"] = float(alpha)
    return cfg


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(lr: float, alpha: float = REGISTERED_ALPHA) -> dict:
    cfg = _cfg_for(lr, alpha)
    require_omp(cfg)
    outdir = _outdir(lr, alpha)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    total = int(cfg["common"]["total_steps"])
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    got = _run_arm_weird(cfg, ARM, "cpu", outdir, seeds, total)
    prov = dict(
        experiment=EXPERIMENT, unregistered=True, unregistered_note=UNREGISTERED,
        created=started, command=sys.argv, elapsed_sec=time.time() - t0,
        python=sys.version, platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, git_hash=_git("rev-parse", "HEAD"),
        git_dirty=_git("status", "--short"),
        arm=ARM, activation="snake", act_alpha=float(alpha),
        lr=float(lr), registered_lr=REGISTERED_LR,
        registered_alpha=REGISTERED_ALPHA,
        changed_axes=(["common.lr_main"] if float(alpha) == REGISTERED_ALPHA
                      else ["arms[SN].dial (act_alpha)"]),
        registered_run="results/comb_isolate_0903",
        registered_outcome=("SN_a1_1216 は lr 0.01 で NUMERIC_DIVERGENCE "
                            "(step 1,000・seed 3)"),
        unregistered_note_for_this_run=_unregistered_note(lr, alpha),
        status=got.get("status"), divergence=got.get("divergence"),
        output_sha256={f"logs/{p.name}": _sha_file(p)
                       for p in sorted((outdir / "logs").glob("*.npz"))})
    (outdir / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[{EXPERIMENT}] lr={lr:g} alpha={alpha:g} -> {got.get('status')} "
          f"in {prov['elapsed_sec']:.1f}s", flush=True)
    return prov


def analyze(lr: float, alpha: float = REGISTERED_ALPHA) -> dict:
    cfg = _cfg_for(lr, alpha)
    outdir = _outdir(lr, alpha)
    if not (outdir / "logs" / f"{ARM}_seed0.npz").exists():
        event = outdir / "arm_status" / f"{ARM}.json"
        got = json.loads(event.read_text(encoding="utf-8")) if event.exists() else {}
        print(json.dumps(dict(status=NUMERIC_DIVERGENCE, **got), ensure_ascii=False))
        return dict(status=NUMERIC_DIVERGENCE, event=got)
    w = _load_new_arm(cfg, outdir, ARM)
    entry = {}
    for key in ("5M", "1M", "early"):
        u = np.asarray(w[key]["u"], dtype=np.float64)
        entry[key] = dict(u=u, onset=_onset_stats(cfg, u))
    cap = _s_cap(cfg, {k: dict(u=entry[k]["u"]) for k in ("early", "1M", "5M")})
    blocks = {a["name"]: a for a in cfg["arms"]}
    unit = _unit_tail(cfg, outdir, blocks[ARM])
    ot = _onset_times(cfg, w["data"]["step"], w["data"]["unfit"])
    result = dict(
        experiment=EXPERIMENT, unregistered=True, lr=float(lr),
        act_alpha=float(alpha), arm=ARM,
        capacity_status=cap["status"],
        n_onset_1m=entry["1M"]["onset"]["n_onset"],
        n_onset_5m=entry["5M"]["onset"]["n_onset"],
        median_log10_U_1m=entry["1M"]["onset"]["median_log10_u"],
        median_log10_U_5m=entry["5M"]["onset"]["median_log10_u"],
        U_5m_seed_values=[float(v) for v in entry["5M"]["u"]],
        k_star_median=float(np.median([r["k_star"] for r in ot["rows"]])),
        n_crossed=int(sum(1 for r in ot["rows"] if not r["censored"])),
        frac_windows_over_median=float(np.median(ot["frac_windows_over"])),
        median_span_all=unit.get("median_span_median_all"),
        median_absv_all=unit.get("median_absv_median_all"),
        median_frozen_abs_frac=unit.get("median_frozen_abs_frac"),
        exact_fit=bool(10 ** float(np.median(np.log10(np.maximum(entry["1M"]["u"], 1e-16)))) <= 1e-8),
        note=_unregistered_note(lr, alpha))
    write_csv(outdir / "diag.csv", [{k: (json.dumps(v, ensure_ascii=False)
                                         if isinstance(v, list) else v)
                                     for k, v in result.items()}])
    (outdir / "summary.md").write_text(
        f"# {EXPERIMENT} — Snake を lr={lr:g}・alpha={alpha:g} で完走させる診断\n\n"
        f"> **{_unregistered_note(lr, alpha)}**\n\n"
        f"- 登録走の `SN_a1_1216`（lr {REGISTERED_LR:g}・alpha {REGISTERED_ALPHA:g}）は "
        f"`NUMERIC_DIVERGENCE`（step 1,000・seed 3）\n"
        f"- 本診断（lr {lr:g}・alpha {alpha:g}）: S-cap `{cap['status']}` / "
        f"発症 1M **{result['n_onset_1m']}/10**・5M **{result['n_onset_5m']}/10** / "
        f"median log10 U (5M) **{result['median_log10_U_5m']:.4f}**\n"
        f"- $k^\\ast$ 中央値 {result['k_star_median']:.0f}・横断 "
        f"{result['n_crossed']}/10・横断窓の割合 "
        f"{result['frac_windows_over_median']:.3f}\n"
        f"- 全ユニット span 中央値 {result['median_span_all']:.4g} / "
        f"|v| 中央値 {result['median_absv_all']:.4g} / "
        f"`frozen_abs` {result['median_frozen_abs_frac']:.4g}\n\n"
        f"**他腕との水準比較には使わない**（lr が違うので同じハーネスではない）。\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--stage", default="run", choices=["run", "analyze"])
    ap.add_argument("--lr", type=float, default=REGISTERED_LR)
    ap.add_argument("--alpha", type=float, default=REGISTERED_ALPHA)
    args = ap.parse_args()
    if args.stage == "run":
        run(args.lr, args.alpha)
    else:
        analyze(args.lr, args.alpha)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""wflip_split_0907 の preflight（本走の前に、記録が壊れていないことだけ見る）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.wflip_split_preflight_0907

見るのは 4 つ:
  1. `layer1_w_flip` が入っていて、形と記録 step が `layer1_w_free` と揃う
  2. 宿主が書く既存列が**バイト同一**（置き直しで壊していない）
  3. S-zbar の 3 分解が実データで通り、変異対照（m=0）が落ちる
  4. S-rank1 が実データで通り、変異対照（u を 1 ビットずらす）が落ちる
本走の判定はしない（短縮走行なので窓が足りない）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import boundary_dense_0907 as BD
from . import edge_law_0905 as E
from . import wflip_split_0907 as W
from . import wflip_split_analyze_0907 as A
from .common import ROOT, load_config

OUT = Path(ROOT) / "results/_preflight_wflip_split_0907"
STEPS = 30_000
SEEDS = [0, 1]
TARGET = 3.041


def _run(module: str, outdir: Path, arm: str) -> None:
    subprocess.run([sys.executable, "-m", module, "--arm", arm, "--steps", str(STEPS),
                    "--seeds", ",".join(str(s) for s in SEEDS), "--outdir", str(outdir)],
                   check=True, cwd=str(ROOT),
                   env={"OMP_NUM_THREADS": "1", "PYTHONPATH": ".", "PATH": "/usr/bin:/bin",
                        "HOME": str(Path.home())})


def main() -> dict:
    tol = load_config(str(W.CONFIG))["analysis"]["tol"]
    mine = OUT / "wflip"
    host = OUT / "host"
    _run("src.wflip_split_0907", mine, "WSref_1216")
    _run("src.boundary_dense_0907", host, "BDref_1216")

    rows = []
    for seed in SEEDS:
        pm = mine / "logs" / f"WSref_1216_seed{seed}.npz"
        ph = host / "logs" / f"BDref_1216_seed{seed}.npz"
        with np.load(pm, allow_pickle=True) as zm, np.load(ph, allow_pickle=True) as zh:
            new = set(zm.files) - set(zh.files)
            same, diff = [], []
            for k in sorted(set(zm.files) & set(zh.files)):
                a, b = np.atleast_1d(zm[k]), np.atleast_1d(zh[k])
                ok = (a.shape == b.shape and a.dtype == b.dtype
                      and (np.array_equal(a, b) if a.dtype.kind not in "fc"
                           else np.array_equal(np.ascontiguousarray(a).view(np.uint8),
                                               np.ascontiguousarray(b).view(np.uint8))))
                (same if ok else diff).append(k)
            wf = zm["layer1_w_flip"]
            wfs = zm["layer1_w_flip_step"].astype(np.int64)
            fre = zm["layer1_w_free_step"].astype(np.int64)
            rows.append(dict(seed=seed, new_columns=sorted(new),
                             n_same=len(same), differing=diff,
                             w_flip_shape=list(wf.shape),
                             steps_match_w_free=bool(np.array_equal(wfs, fre))))
        d = A.load_seed(mine / "logs", "WSref_1216", seed)
        zb = A.s_zbar(d, TARGET)
        zbm = A.s_zbar(d, TARGET, drop_m=True)
        r1 = A.s_rank1(d, TARGET)
        r1m = A.s_rank1(d, TARGET, mutant=True)
        rows[-1].update(
            s_zbar=float(np.nanmedian(zb)), s_zbar_max=float(np.nanmax(zb)),
            s_zbar_mutant=float(np.nanmedian(zbm)),
            rho=float(np.nanmedian(r1["rho"])), rho_max=float(np.nanmax(r1["rho"])),
            rho_mutant=float(np.nanmedian(r1m["rho"])),
            rel=float(np.nanmedian(r1["rel"])), rel_mutant=float(np.nanmedian(r1m["rel"])))

    ok = all(
        r["differing"] == ["arm", "run_id"]      # 腕名の文字列だけが違ってよい
        and set(r["new_columns"]) == {"layer1_w_flip", "layer1_w_flip_step"}
        and r["steps_match_w_free"]
        and r["s_zbar_max"] < float(tol["zbar_rel"])
        and r["s_zbar_mutant"] > float(tol["zbar_mutant_min"])
        and r["rho"] < float(tol["rank1_rho_max"])
        and r["rho_mutant"] > float(tol["rank1_rho_mutant_min"])
        for r in rows)
    res = dict(pass_=bool(ok), steps=STEPS, seeds=SEEDS, rows=rows,
               tol={k: float(v) for k, v in tol.items()},
               git_head=E._git_head())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    return res


if __name__ == "__main__":
    raise SystemExit(0 if main()["pass_"] else 1)

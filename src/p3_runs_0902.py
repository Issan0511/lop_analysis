"""p3_runs_0902: 現象3 の 2 走 — 延長 15M（p3_extend_0902）と谷埋め介入（valley_clamp_0902）。

宿主は gate_dial_0902（d059d4a）。config は configs/gate_dial_0902.yaml を読み、腕表と
総 step だけをここで足す。gate_dial の validate_config は 5M・14 腕に固定されているので
通さない。したがって凍結した設計からの差分は **このファイルの EXPS だけ** である。
事前登録は vault `可塑性喪失/論点/現象3_非ReLU戻り道の対応づけ_0902.md` §7。

    OMP_NUM_THREADS=1 .venv/bin/python -m src.p3_runs_0902 --exp extend --arm E_1216
    OMP_NUM_THREADS=1 .venv/bin/python -m src.p3_runs_0902 --exp clamp  --arm Gc_b1_1216
    OMP_NUM_THREADS=1 .venv/bin/python -m src.p3_runs_0902 --exp extend --s-ext E_1216

延長走の腕: 既存の dial 腕（S_b1 / S_b0p3 / G_b0p3）は登録済みの腕ブロックを逐語で再利用し、
対照 2 腕（E_1216 = ELU α=1 / LR_1216 = leaky a=0.1）は同じ雛形で新設する。S-ext は先頭の
記録が committed ログ（gate_dial_0902 / gate_dose_0830）と bit 一致することを検査する。
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import ROOT, load_config
from .gate_dial_0902 import CONFIG, _run_arm
from .mlp2_phase0 import require_omp

TEMPLATE = dict(stage=0, hidden=[100], centered_layers=[1],
                target_mu_norm=3.041, target_dose=12.16)
DIAL_LOGS = "results/gate_dial_0902/logs"
DOSE_LOGS = "results/gate_dose_0830/logs"
EXPS = {
    "extend": dict(
        outdir="results/p3_extend_0902", total=15_000_000,
        checkpoints=[0, 1_000_000, 5_000_000, 10_000_000, 15_000_000],
        arms={
            "E_1216": dict(family="elu", activation="elu", dial=1.0,
                           u_star=None, u_fr=13.8155),
            "LR_1216": dict(family="leaky", activation="leaky", dial=0.1,
                            u_star=None, u_fr=None),
            "S_b1_1216": None, "S_b0p3_1216": None, "G_b0p3_1216": None,
        },
        reference={"E_1216": DOSE_LOGS, "LR_1216": DOSE_LOGS,
                   "S_b1_1216": DIAL_LOGS, "S_b0p3_1216": DIAL_LOGS,
                   "G_b0p3_1216": DIAL_LOGS}),
    "clamp": dict(
        outdir="results/valley_clamp_0902", total=5_000_000,
        checkpoints=[0, 1_000_000, 5_000_000],
        arms={
            "Gc_b1_1216": dict(family="gelu_clamp", activation="gelu_clamp",
                               dial=1.0, u_star=0.7519, u_fr=5.394),
            "Sc_b3_1216": dict(family="silu_clamp", activation="silu_clamp",
                               dial=3.0, u_star=0.4262, u_fr=5.520),
        },
        reference={}),
    "clamp0": dict(
        outdir="results/valley_clamp0_0902", total=5_000_000,
        checkpoints=[0, 1_000_000, 5_000_000],
        arms={
            "Gz_b1_1216": dict(family="gelu_clamp0", activation="gelu_clamp0",
                               dial=1.0, u_star=0.7519, u_fr=5.394),
            "Sz_b3_1216": dict(family="silu_clamp0", activation="silu_clamp0",
                               dial=3.0, u_star=0.4262, u_fr=5.520),
        },
        reference={}),
}


def build_cfg(exp: str) -> dict:
    c = copy.deepcopy(load_config(str(CONFIG)))
    E = EXPS[exp]
    c["common"]["total_steps"] = int(E["total"])
    c["common"]["checkpoints"] = list(E["checkpoints"])
    c["activation"].setdefault("gelu_clamp", {"name": "gelu_clamp"})
    c["activation"].setdefault("silu_clamp", {"name": "silu_clamp"})
    c["activation"].setdefault("gelu_clamp0", {"name": "gelu_clamp0"})
    c["activation"].setdefault("silu_clamp0", {"name": "silu_clamp0"})
    have = {a["name"] for a in c["arms"]}
    for name, spec in E["arms"].items():
        if spec is None:
            if name not in have:
                raise ValueError(f"{name} is not a registered gate_dial arm")
            continue  # 登録済みの腕ブロックを逐語で使う
        if name in have:
            raise ValueError(f"{name} already exists in the host config")
        c["arms"].append(dict(name=name, **TEMPLATE, **spec))
    return c


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_arm(exp: str, arm: str, steps: int | None, outdir: Path | None) -> dict:
    cfg = build_cfg(exp)
    require_omp(cfg)
    E = EXPS[exp]
    total = int(steps) if steps else int(E["total"])
    out = outdir or (Path(ROOT) / E["outdir"])
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    done = out / "arm_status" / f"{arm}_done.json"
    if done.exists() and json.loads(done.read_text())["total_steps"] == total:
        print(f"[{arm}] done marker found; nothing to do", flush=True)
        return json.loads(done.read_text())
    t0 = time.time()
    head_launch = _git_head()
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "src"],
                                        cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        dirty = "unknown"
    res = _run_arm(cfg, arm, "cpu", out, seeds, total)
    status = dict(exp=exp, arm=arm, total_steps=total, seeds=seeds,
                  status=res.get("status"), elapsed_sec=res.get("elapsed_sec"),
                  wall_sec=time.time() - t0, git_head=_git_head(),
                  git_head_at_launch=head_launch, src_dirty_at_launch=dirty,
                  arm_block=next(a for a in cfg["arms"] if a["name"] == arm))
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def s_ext(exp: str, arm: str, outdir: Path | None) -> dict:
    """先頭の共通記録が参照ログと bit 一致するか（state_hash_1m も含む）。"""
    E = EXPS[exp]
    out = (outdir or (Path(ROOT) / E["outdir"])) / "logs"
    ref = Path(ROOT) / E["reference"][arm]
    rows = []
    for seed in range(10):
        pa, pb = out / f"{arm}_seed{seed}.npz", ref / f"{arm}_seed{seed}.npz"
        if not (pa.exists() and pb.exists()):
            rows.append(dict(seed=seed, status="MISSING"))
            continue
        a = np.load(pa, allow_pickle=True)
        b = np.load(pb, allow_pickle=True)
        n = min(len(a["step"]), len(b["step"]))
        bad = {}
        for k in sorted(set(a.files) & set(b.files)):
            if k in ("state_hash_final", "state_hash_1m"):
                continue  # final は延長で当然違う。1M のハッシュは下で別に見る
            x, y = a[k], b[k]
            if x.ndim >= 1 and x.shape[0] == len(a["step"]):
                x = x[:n]
            if y.ndim >= 1 and y.shape[0] == len(b["step"]):
                y = y[:n]
            if x.shape != y.shape:
                bad[k] = f"shape {x.shape} vs {y.shape}"
            elif x.dtype.kind in "fiub":
                if not np.array_equal(x, y, equal_nan=True):
                    bad[k] = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
            elif not np.array_equal(x, y):
                bad[k] = "differs"
        h = (str(a["state_hash_1m"]) == str(b["state_hash_1m"])) if n > 1000 else None
        rows.append(dict(seed=seed, status="OK", n_records=int(n), n_bad=len(bad),
                         bad=bad, state_hash_1m_equal=h))
    passed = all(r.get("status") == "OK" and r["n_bad"] == 0
                 and r["state_hash_1m_equal"] in (None, True) for r in rows)
    result = dict(exp=exp, arm=arm, pass_=passed, rows=rows)
    path = out.parent / "sanity" / f"s_ext_{arm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[S-ext {arm}] {'PASS' if passed else 'FAIL'} -> {path}", flush=True)
    for r in rows:
        print("  ", r, flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=sorted(EXPS))
    p.add_argument("--arm")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--s-ext", default=None, metavar="ARM")
    a = p.parse_args()
    out = Path(a.outdir).resolve() if a.outdir else None
    if a.s_ext:
        s_ext(a.exp, a.s_ext, out)
        return
    if not a.arm or a.arm not in EXPS[a.exp]["arms"]:
        p.error(f"--arm must be one of {sorted(EXPS[a.exp]['arms'])}")
    run_arm(a.exp, a.arm, a.steps, out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Checks of cifar5p1_mlp_snfix_1019 (spec §6).  Each check carries a deliberately broken variant
that it has to catch.

    python3 src/cifar5p1_mlp_snfix_1019_checks.py --out results/cifar5p1_mlp_snfix_1019/checks

  S-fixed           SN06 / SN3: phi, phi' against float64 autograd; bit-identical to SnakeFamily's snake
                    branch at the same alpha; alpha constant through training (mutant: alpha that drifts)
  S-frz             SNAfrz1 equals SNA through task 1, freezes alpha at SNA's task-1 value, keeps it, and
                    its fresh control equals SNA's (mutant: the freeze one task late)
  S-graph           CUDA graph == eager for the three new arms (mutant: eager at a 2^-20 larger lr)
  S-nochange-early  R and SNA, tasks 1-2, rerun here == the committed 0920 rows (mutant: R at a 2^-20 larger lr)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                    # noqa: E402
from src import rlcifar_mlp_battle_0918 as B        # noqa: E402
from src import cifar5p1_mlp_0920 as C              # noqa: E402
from src import cifar5p1_mlp_snfix_1019 as S        # noqa: E402

EPS32 = 2.0 ** -24
QUIET = (lambda m: None)


def rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def strip_arm(rs: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "arm"} for r in rs]


def run(arm, out: Path, device, cifar, n_tasks, fresh=False, graph=True, lr=S.LR, seeds=S.SEEDS):
    C.run(arm, list(seeds), S.COND, n_tasks, device, out, lr=lr, cifar=cifar, fresh=fresh, graph=graph,
          progress=QUIET)
    return S.LAST_ACT[arm]


class DriftSnake(S.FixedSnake):
    """Mutant: a 'fixed' Snake whose alpha follows the width like SNA's (c / W, clipped)."""

    @torch.no_grad()
    def update(self, z1, z2):
        super().update(z1, z2)
        for l in (0, 1):
            self.A[l].copy_((0.6 / self.Vd[l].sqrt()).clamp(0.005, 3.0))


def with_arm(name: str, factory):
    """Temporarily resolve one extra arm name in C.make_act (check-only arms)."""
    inner = C.make_act

    def make(arm, *a, **k):
        if arm == name:
            act = factory()
            S.LAST_ACT[arm] = act
            return act
        return inner(arm, *a, **k)
    return inner, make


# --------------------------------------------------------------------------

def check_fixed(device, cifar, work: Path) -> dict:
    g = torch.Generator(device="cpu").manual_seed(1019)
    out = {}
    ok = True
    for arm, alpha in S.FIXED.items():
        act = S.FixedSnake(arm, alpha)
        act.init_state(2, device, "check")
        z64 = (3.0 * torch.randn(2, 64, C.HIDDEN, generator=g, dtype=torch.float64)).to(device)
        a64 = torch.full((2, 1, C.HIDDEN), float(torch.tensor(alpha, dtype=torch.float32)), dtype=torch.float64,
                         device=device)
        zz = z64.clone().requires_grad_(True)
        y = zz + torch.sin(a64 * zz) ** 2 / a64
        d_auto = torch.autograd.grad(y.sum(), zz)[0]
        d_form = 1.0 + torch.sin(2.0 * a64 * z64)
        err_auto = float((d_auto - d_form).abs().max())
        z32 = z64.float()
        phi_err = ((act.phi(z32, 0).double() - y.detach()).abs()
                   / (8 * EPS32 * (z64.abs() + 2.0 / a64 + 1.0))).max()
        dphi_err = ((act.dphi(z32, 0).double() - d_form).abs()
                    / (8 * EPS32 * (2.0 + 2.0 * a64 * z64.abs()))).max()
        sf = B.SnakeFamily("SNA", "snake", c=alpha, beta=0.01, lo=0.005, hi=3.0)
        sf.init_state(2, device, "check")             # V = 1, so SNA's alpha is clip(alpha / 1) = alpha
        same_alpha = all(torch.equal(sf.alpha(l), act.alpha(l)) for l in (0, 1))
        same_phi = all(torch.equal(sf.phi(z32, l), act.phi(z32, l)) and torch.equal(sf.dphi(z32, l), act.dphi(z32, l))
                       for l in (0, 1))
        trained = run(arm, work / f"fixed_{arm}", device, cifar, 2, seeds=[0, 1])
        a32 = float(torch.tensor(alpha, dtype=torch.float32))
        const = all(bool((t == a32).all()) for t in trained.A)
        stat_rows = rows(work / f"fixed_{arm}" / "per_task.csv")
        const_rows = all(float(r[f"alpha_{q}_l{l}"]) == a32 for r in stat_rows for q in ("med", "min", "max")
                         for l in (1, 2))
        res = {"autograd_vs_formula_f64": err_auto, "phi_f32_over_bound": float(phi_err),
               "dphi_f32_over_bound": float(dphi_err), "same_as_SnakeFamily_at_alpha": same_alpha and same_phi,
               "alpha_constant_after_2_tasks": const, "alpha_constant_in_rows": const_rows}
        res["ok"] = (err_auto < 1e-12 and phi_err <= 1.0 and dphi_err <= 1.0 and same_alpha and same_phi
                     and const and const_rows)
        ok &= res["ok"]
        out[arm] = res
    inner, make = with_arm("SN06drift", lambda: DriftSnake("SN06drift", 0.6))
    C.make_act = make
    try:
        drift = run("SN06drift", work / "fixed_mut", device, cifar, 2, seeds=[0, 1])
    finally:
        C.make_act = inner
    detected = not all(bool((t == float(torch.tensor(0.6, dtype=torch.float32))).all()) for t in drift.A)
    return {"pass": ok, "per_arm": out, "mutants": {"alpha_drifts_like_SNA": {"detected": detected}}}


def check_frz(device, cifar, work: Path) -> dict:
    n_tasks = 3                                        # hard, easy, hard; the fresh control runs on task 3
    sna = run("SNA", work / "frz_SNA", device, cifar, n_tasks, fresh=True)
    cap = run("SNAcap", work / "frz_SNAcap", device, cifar, n_tasks, fresh=True)
    frz = run("SNAfrz1", work / "frz_SNAfrz1", device, cifar, n_tasks, fresh=True)
    r_sna, r_cap, r_frz = (rows(work / f"frz_{a}" / "per_task.csv") for a in ("SNA", "SNAcap", "SNAfrz1"))
    f_sna, f_cap, f_frz = (rows(work / f"frz_{a}" / "fresh_control.csv") for a in ("SNA", "SNAcap", "SNAfrz1"))
    t1 = lambda rs: [r for r in rs if r["task"] == "1"]
    cap_is_sna = strip_arm(r_cap) == strip_arm(r_sna) and strip_arm(f_cap) == strip_arm(f_sna)
    task1_same = strip_arm(t1(r_frz)) == strip_arm(t1(r_sna))
    E = S.FREEZE_AFTER
    frozen_is_task1 = all(torch.equal(frz.captured[E][l], cap.captured[E][l]) for l in (0, 1))
    kept = all(torch.equal(frz.captured[2 * E][l], frz.captured[E][l]) for l in (0, 1))
    sna_moves = not all(torch.equal(cap.captured[2 * E][l], cap.captured[E][l]) for l in (0, 1))
    stat_cols = [f"alpha_{q}_l{l}" for q in ("med", "min", "max") for l in (1, 2)]
    by_task = {(r["slot"], r["task"]): r for r in r_frz}
    rows_const = all(by_task[(s, t)][c] == by_task[(s, "1")][c] for s in {r["slot"] for r in r_frz}
                     for t in ("2", "3") for c in stat_cols)
    later_differs = strip_arm([r for r in r_frz if r["task"] != "1"]) != strip_arm([r for r in r_sna if r["task"] != "1"])
    fresh_same = strip_arm(f_frz) == strip_arm(f_sna)
    ok = cap_is_sna and task1_same and frozen_is_task1 and kept and sna_moves and rows_const and later_differs and fresh_same
    # mutant: the freeze one task late
    inner, make = with_arm("SNAfrz2", lambda: S.FrozenSnake("SNAfrz2", freeze_after=2 * E))
    C.make_act = make
    try:
        late = run("SNAfrz2", work / "frz_mut", device, cifar, n_tasks, fresh=False)
    finally:
        C.make_act = inner
    detected = not all(torch.equal(late.captured[2 * E][l], cap.captured[E][l]) for l in (0, 1))
    return {"pass": ok, "frozenSnake_without_freeze_is_SNA": cap_is_sna, "task1_rows_equal_SNA": task1_same,
            "frozen_alpha_is_SNA_task1_alpha": frozen_is_task1, "alpha_kept_at_task3_start": kept,
            "SNA_alpha_moves_after_task1": sna_moves, "alpha_stats_constant_in_rows": rows_const,
            "later_rows_differ_from_SNA": later_differs, "fresh_equals_SNA_fresh": fresh_same,
            "mutants": {"freeze_one_task_late": {"detected": detected}}}


def check_graph(device, cifar, work: Path) -> dict:
    res, ok = {}, True
    for arm in S.NEW_ARMS:
        run(arm, work / f"g_{arm}", device, cifar, 2, fresh=True, graph=True)
        run(arm, work / f"e_{arm}", device, cifar, 2, fresh=True, graph=False)
        same = all(rows(work / f"g_{arm}" / f) == rows(work / f"e_{arm}" / f)
                   for f in ("per_task.csv", "fresh_control.csv"))
        res[arm] = same
        ok &= same
    run("SN3", work / "e_mut", device, cifar, 2, fresh=True, graph=False, lr=S.LR * (1 + 2.0 ** -20))
    base = rows(work / "g_SN3" / "per_task.csv")
    mut = [{**r, "lr": base[0]["lr"]} for r in rows(work / "e_mut" / "per_task.csv")]   # the lr column itself is not the signal
    detected = mut != base
    return {"pass": ok, "graph_equals_eager": res, "mutants": {"eager_lr_2^-20_larger": {"detected": detected}}}


def check_nochange_early(device, cifar, work: Path) -> dict:
    res, ok = {}, True
    for arm in ("R", "SNA"):
        want = [r for r in rows(S.HOST_RESULTS / S.cell_name(arm) / "per_task.csv") if r["task"] in ("1", "2")]
        run(arm, work / f"n_{arm}", device, cifar, 2, fresh=False)
        got = rows(work / f"n_{arm}" / "per_task.csv")
        same = len(want) == 20 and got == want
        res[arm] = {"rows": len(got), "identical_to_committed_0920": same}
        ok &= same
    want = [r for r in rows(S.HOST_RESULTS / S.cell_name("R") / "per_task.csv") if r["task"] in ("1", "2")]
    run("R", work / "n_mut", device, cifar, 2, fresh=False, lr=S.LR * (1 + 2.0 ** -20))
    got_lr = [{**r, "lr": want[0]["lr"]} for r in rows(work / "n_mut" / "per_task.csv")]
    detected = got_lr != want
    return {"pass": ok, **res, "mutants": {"R_lr_2^-20_larger": {"detected": detected}}}


def all_pass(res: dict) -> bool:
    ok = True
    for r in res.values():
        ok &= bool(r.get("pass"))
        for m in r.get("mutants", {}).values():
            ok &= m.get("detected") is True
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    torch.set_num_threads(S.THREADS)
    device = H.setup(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="snfix_checks_", dir=str(out)))
    t0 = time.time()
    cifar = C.Cifar100()
    plan = [("S-fixed", check_fixed), ("S-frz", check_frz), ("S-graph", check_graph),
            ("S-nochange-early", check_nochange_early)]
    res = {}
    for name, fn in plan:
        t1 = time.time()
        try:
            r = fn(device, cifar, work)
        except Exception as ex:
            r = {"pass": False, "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
        r["seconds"] = round(time.time() - t1, 1)
        res[name] = r
        mut = {k: v.get("detected") for k, v in r.get("mutants", {}).items()}
        print(f"[{time.strftime('%T')}] {name:17s} pass={r.get('pass')} mutants={mut} ({r['seconds']}s)"
              + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    summary = {"scope": "implementation checks only; no performance verdict", "all_pass": all_pass(res),
               "checks": res, "git": {**B.git_state(), "dirty_src_analysis_specs": S.git_dirty()},
               "torch": torch.__version__, "device": str(device), "torch_threads": torch.get_num_threads(),
               "spec": S.SPEC, "seconds": round(time.time() - t0, 1)}
    (out / "checks.json").write_text(json.dumps(summary, indent=2, default=str))
    shutil.rmtree(work, ignore_errors=True)
    print(f"all_pass={summary['all_pass']} -> {out / 'checks.json'}", flush=True)


if __name__ == "__main__":
    main()

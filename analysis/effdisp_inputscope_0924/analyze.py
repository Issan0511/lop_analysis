"""Read-only input-scope analysis. A row labelled task t is W[t-1] -> W[t].

The actual decomposition uses the *current* input bank for Q and X:
Vnext(current)-Vprev(previous) = Q(current)+2X(current)+G_pre.
Reference P1--P4 remain the earlier fixed-bank definitions.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis.effdisp_validation_0924.metrics import summarize_task_table  # noqa: E402; registered fixed-reference rule

SEEDS = tuple(range(200, 205))
CELLS = ("X0Y0", "X0Y1", "X1Y0", "X1Y1")
SPEC_NAMES = ("spec_effdisp_inputscope_0924.md", "spec_effdisp_inputscope_0924_addendum_c06.md")
DEFAULT_RAW = Path("/home/issan/Projects/obsidian-research-data/effdisp_inputscope_0924/raw")
_BANK_CACHE: dict[str,dict[str,np.ndarray]] = {}


@dataclass(frozen=True)
class Series:
    environment: str
    arm: str
    activation: str
    cell: str
    seed: int
    tasks: int
    path: Path


def expected_series(raw_root: Path) -> list[Series]:
    result = []
    for act in ("LR", "SNA06"):
        for k in (1, 7):
            for cell in CELLS:
                arm = f"{act}_k{k}_{cell}"
                result += [Series("conda", arm, act, cell, seed, 400,
                                  raw_root / "conda" / arm / f"seed{seed}") for seed in SEEDS]
    for act in ("LR", "SNA06", "SN05", "SNA03"):
        for cell in (CELLS if act in ("LR", "SNA06") else ("X0Y1",)):
            tasks = 150 if act in ("LR", "SNA06") and cell == "X0Y1" else 50
            arm = f"{act}_{cell}"
            result += [Series("mnist", arm, act, cell, seed, tasks,
                              raw_root / "mnist" / arm / f"seed_{seed}") for seed in SEEDS]
    assert len(result) == 130
    return result


def _snapshot(s: Series, task: int) -> Path:
    return s.path / (f"t{task:03d}.npz" if s.environment == "conda" else f"state_{task:03d}.npz")


def _json(path: Path):
    return json.loads(path.read_text())


def _write_gzip_csv(table: pd.DataFrame, path: Path) -> dict:
    """Write reproducible gzip CSV and verify its uncompressed SHA256."""
    temporary=path.with_name(path.name+".tmp")
    digest=hashlib.sha256()

    class Writer:
        def __init__(self, stream):
            self.stream=stream

        def write(self, value: str) -> int:
            data=value.encode("utf-8")
            digest.update(data)
            self.stream.write(data)
            return len(value)

    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as compressed:
            table.to_csv(Writer(compressed),index=False,lineterminator="\n")
    temporary.replace(path)
    expanded=hashlib.sha256()
    with gzip.open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            expanded.update(chunk)
    if expanded.digest()!=digest.digest():
        raise RuntimeError(f"gzip round-trip SHA256 mismatch: {path}")
    return {"path":str(path),"compression":"gzip","mtime":0,"gzip_filename":"",
            "compressed_bytes":path.stat().st_size,
            "compressed_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "uncompressed_sha256":digest.hexdigest(),"round_trip_verified":True}


def _npz(path: Path, *required: str, optional: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        absent = set(required) - set(data.files)
        if absent:
            raise ValueError(f"{path}: missing keys {sorted(absent)}")
        return {k: np.asarray(data[k]) for k in (*required, *(k for k in optional if k in data))}


def _spec_hashes() -> dict[str, str]:
    return {f"specs/{name}": hashlib.sha256((REPO / "specs" / name).read_bytes()).hexdigest()
            for name in SPEC_NAMES}


def preflight(s: Series) -> dict:
    """Verify schema and completion without reading scientific outcomes."""
    missing = [_snapshot(s, t) for t in range(s.tasks + 1) if not _snapshot(s, t).is_file()]
    if s.environment == "mnist":
        missing += [p for p in [s.path / "input_bank.npz", s.path / "per_task.json",
                                *(s.path / f"task_{t:03d}_input.npz" for t in range(1, s.tasks + 1))]
                    if not p.is_file()]
    meta_path = s.path.parent / "metadata.json" if s.environment == "mnist" else s.path.parents[1] / "metadata.json"
    status_path = s.path.parent / "status.json" if s.environment == "mnist" else s.path / "status.json"
    reason, status, meta = "", "INCOMPLETE", {}
    if not meta_path.is_file() or not status_path.is_file():
        reason = "missing metadata/status"
    else:
        meta, all_status = _json(meta_path), _json(status_path)
        run_status = all_status.get(str(s.seed), {}) if s.environment == "mnist" else all_status
        if s.environment == "conda":
            design = {a["name"]: a for a in meta.get("arms", [])}.get(s.arm)
            expected_k=int(s.arm.split("_k",1)[1].split("_",1)[0])
            if (design is None or design.get("activation") != ("leaky" if s.activation=="LR" else "sna06")
                or design.get("X") != int(s.cell[1]) or design.get("Y") != int(s.cell[3])
                or design.get("k") != expected_k):
                reason = "arm absent or activation mismatch"
            elif (meta.get("args", {}).get("tasks") != 400 or meta.get("args", {}).get("period") != 10000
                  or meta.get("m") != 20 or meta.get("r") != 5 or s.seed not in meta.get("args", {}).get("seeds", SEEDS)):
                reason = "metadata tasks/period/seeds mismatch"
            elif run_status.get("status") != "COMPLETED" or run_status.get("last_saved_task") != s.tasks:
                status = run_status.get("status", "INCOMPLETE")
                reason = f"runner status={status} task={run_status.get('last_saved_task')}"
        else:
            if (meta.get("mode") != "inputscope_mnist" or meta.get("activation") != s.activation
                    or meta.get("cell") != s.cell or meta.get("tasks") != s.tasks
                    or meta.get("steps_per_task") != 30000 or meta.get("optimizer", "adam").lower() != "adam"
                    or meta.get("epochs") != 400 or meta.get("batch") != 16
                    or meta.get("n_images") != 1200 or meta.get("lr") != .001
                    or meta.get("dims") != [784,100,100,10]
                    or s.seed not in meta.get("seeds", SEEDS) or meta.get("smoke")):
                reason = "metadata differs from registered suite"
            elif run_status.get("state") != "complete" or run_status.get("completed_tasks") != s.tasks:
                status = run_status.get("state", "INCOMPLETE").upper()
                reason = f"runner status={status} tasks={run_status.get('completed_tasks')}"
        source_hashes = meta.get("spec_sha256", {})
        if source_hashes and source_hashes != _spec_hashes():
            reason = "source spec hashes differ from current registered specs"
    if missing and not reason:
        reason = f"{len(missing)} required files missing"
    if not s.path.exists():
        status = "MISSING"
    return {"environment": s.environment, "arm": s.arm, "seed": s.seed,
            "status": "COMPLETE" if not reason and not missing else status,
            "reason": reason, "missing_count": len(missing),
            "first_missing": str(missing[0]) if missing else "",
            "metadata_path": str(meta_path), "status_path": str(status_path),
            "source_git_hash": meta.get("git_hash", "") if meta_path.is_file() else "",
            "source_spec_sha256": json.dumps(meta.get("spec_sha256", {}),sort_keys=True)
            if meta_path.is_file() else ""}


def _support32(mu: np.ndarray) -> np.ndarray:
    bits = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.float64)
    return np.column_stack((np.broadcast_to(np.asarray(mu[:15], dtype=np.float64), (32, 15)), bits))


def _moments(projected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = projected.mean(axis=1)
    centered = projected - mean[:, None]
    return mean, centered


def paired_terms(w0: np.ndarray, w1: np.ndarray, b0: np.ndarray, b1: np.ndarray,
                 x_previous: np.ndarray, x_current: np.ndarray) -> dict[str, np.ndarray]:
    """Exact per-unit same-ID preactivation decomposition, population means."""
    w0, w1 = np.asarray(w0, dtype=np.float64), np.asarray(w1, dtype=np.float64)
    b0, b1 = np.asarray(b0, dtype=np.float64), np.asarray(b1, dtype=np.float64)
    xp, xc = np.asarray(x_previous, dtype=np.float64), np.asarray(x_current, dtype=np.float64)
    if xp.shape != xc.shape or xp.ndim != 2 or xp.shape[1] != w0.shape[1] or w0.shape != w1.shape:
        raise ValueError("paired supports/weights have incompatible shapes")
    a = (w1 - w0) @ xc.T + (b1 - b0)[:, None]
    e = (np.zeros_like(a) if np.array_equal(xc,xp) else w0 @ (xc - xp).T)
    return _paired_from_a_e(a,e)


def _paired_from_a_e(a: np.ndarray, e: np.ndarray) -> dict[str,np.ndarray]:
    mean_a, ac = _moments(a)
    mean_e, ec = _moments(e)
    q = np.mean(ac * ac, axis=1)
    update = np.mean(a * a, axis=1)
    input_change = np.mean(e * e, axis=1)
    cross = 2 * np.mean(a * e, axis=1)
    total = np.mean((a + e) ** 2, axis=1)
    return {"Q_current": q, "mean_A_sq": mean_a**2, "A_energy": update,
            "E_energy": input_change, "AE_cross": cross, "total_energy": total,
            "mean_E_sq": mean_e**2, "A_identity_residual": update - q - mean_a**2,
            "total_identity_residual": total - update - input_change - cross,
            "mean_A": mean_a, "mean_E": mean_e,
            "centered_E_energy": np.mean(ec * ec, axis=1)}


def _projected_terms(p0: np.ndarray, p1: np.ndarray) -> dict[str, np.ndarray]:
    m0, c0 = _moments(p0)
    _, c1 = _moments(p1)
    d = c1 - c0
    v = np.mean(c0*c0, axis=1)
    q = np.mean(d*d, axis=1)
    x = np.mean(c0*d, axis=1)
    vn = np.mean(c1*c1, axis=1)
    return {"V": v, "Q": q, "X": x, "V_next": vn,
            "D_norm": np.sqrt(np.maximum(q, 0)), "input_mean": m0,
            "identity_residual": vn-v-q-2*x}


def _raw_terms(w0: np.ndarray, w1: np.ndarray) -> dict[str, np.ndarray]:
    d = w1-w0
    v = np.sum(w0*w0, axis=1)
    q = np.sum(d*d, axis=1)
    x = np.sum(w0*d, axis=1)
    vn = np.sum(w1*w1, axis=1)
    return {"V": v, "Q": q, "X": x, "V_next": vn,
            "D_norm": np.sqrt(q), "identity_residual": vn-v-q-2*x}


def actual_terms(w0: np.ndarray, w1: np.ndarray, x_previous: np.ndarray,
                 x_current: np.ndarray) -> dict[str, np.ndarray]:
    """Per-unit current-covariance Q/X and pre-update covariance drift G_pre."""
    previous = np.asarray(w0, dtype=np.float64) @ np.asarray(x_previous, dtype=np.float64).T
    current0 = np.asarray(w0, dtype=np.float64) @ np.asarray(x_current, dtype=np.float64).T
    current1 = np.asarray(w1, dtype=np.float64) @ np.asarray(x_current, dtype=np.float64).T
    current = _projected_terms(current0, current1)
    old = _projected_terms(previous, previous)["V"]
    g = current["V"] - old
    return {**current, "V": old, "Vprev_current": current["V"], "G_pre": g,
            "identity_residual": current["V_next"] - old - current["Q"] - 2*current["X"] - g}


def _sum_terms(unit: dict[str, np.ndarray]) -> dict[str, float]:
    """Sum units; actual c/rho use current-bank Vprev_current, not Vprev_old."""
    result = {k: float(np.sum(v, dtype=np.float64)) for k, v in unit.items()
              if k not in ("D_norm", "input_mean")}
    v, q, x = result["V"], result["Q"], result["X"]
    geometry_v=result.get("Vprev_current",v)
    result.update(D_norm=math.sqrt(max(q, 0)),
                  c=x/math.sqrt(geometry_v*q) if geometry_v*q > 0 else math.nan,
                  rho=math.sqrt(q/geometry_v) if geometry_v > 0 else math.nan,
                  balance=-2*x/q if q > 0 else math.nan)
    return result


def _window_specs(s: Series) -> dict[str, tuple[int, int]]:
    if s.environment == "conda":
        return {"secondary100": (81, 100), "primary400": (321, 400)}
    if s.tasks == 150:
        return {"primary50": (41, 50), "long150": (121, 150)}
    return {"primary50": (41, 50)}


def _diagnostics(s: Series) -> dict[int, dict]:
    if s.environment == "conda":
        return {}
    rows = _json(s.path / "per_task.json")
    by_task = {int(row["task"]): row for row in rows}
    if set(by_task) != set(range(1, s.tasks+1)):
        raise ValueError("per_task.json task set mismatch")
    return by_task


def _validated_mnist_bank(path: Path) -> dict[str,np.ndarray]:
    bank=_npz(path,"images","mean","covariance","subset_idx")
    digest=hashlib.sha256()
    for field in ("images","mean","covariance","subset_idx"):
        digest.update(np.ascontiguousarray(bank[field]).view(np.uint8))
    key=digest.hexdigest()
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    images=bank["images"].astype(np.float64)
    if images.shape!=(1200,784) or not np.all(np.isfinite(images)):
        raise ValueError("input bank images must be 1200x784 finite")
    mean=images.mean(axis=0)
    centered=images-mean
    covariance=centered.T@centered/len(images)
    if not np.allclose(mean,bank["mean"],rtol=2e-5,atol=2e-6):
        raise ValueError("base bank mean mismatch")
    if not np.allclose(covariance,bank["covariance"],rtol=3e-5,atol=3e-6):
        raise ValueError("base bank covariance mismatch")
    bank["images"]=images
    _BANK_CACHE[key]=bank
    return bank


def _inputs(s: Series, initial: dict, task: int, current: dict,
            previous_input: np.ndarray | None, bank: dict | None) -> tuple[np.ndarray, np.ndarray, str, str]:
    if s.environment == "conda":
        xcur = _support32(current["mu_input"])
        xprev = _support32(initial["mu_input"]) if task == 1 else previous_input
        return xprev, xcur, str(_snapshot(s, max(0,task-1))), str(_snapshot(s, task))
    inp_path = s.path / f"task_{task:03d}_input.npz"
    inp = _npz(inp_path, "permutation", "labels")
    perm = inp["permutation"].astype(np.int64)
    images = bank["images"].astype(np.float64, copy=False)
    if perm.shape != (784,) or np.unique(perm).size != 784 or perm.min() != 0 or perm.max() != 783:
        raise ValueError(f"{inp_path}: permutation invalid")
    if inp["labels"].shape != (1200,) or np.any((inp["labels"]<0)|(inp["labels"]>9)):
        raise ValueError(f"{inp_path}: labels invalid")
    xcur = images[:, perm]
    xprev = xcur if task == 1 else previous_input
    return xprev, xcur, str(s.path / f"task_{max(1,task-1):03d}_input.npz"), str(inp_path)


def transition_rows(s: Series) -> tuple[list[dict], list[dict]]:
    """One complete seed; no writes. Reference, actual, raw and paired-bank rows."""
    previous = _npz(_snapshot(s, 0), *(('W','b','mu_input','mu_original','teacher32targets')
                                      if s.environment == 'conda' else ('W1','b1')))
    initial = previous
    bank = (_validated_mnist_bank(s.path / "input_bank.npz")
            if s.environment == "mnist" else None)
    if bank is not None:
        xref = bank["images"]
    else:
        xref = _support32(initial["mu_input"])
    diagnostics = _diagnostics(s)
    rows, unit_rows = [], []
    windows = _window_specs(s)
    accum: dict[tuple[str,str], dict[str,np.ndarray]] = {}
    x_previous = None
    for task in range(1,s.tasks+1):
        current_path = _snapshot(s,task)
        if s.environment == "conda":
            current = _npz(current_path,"W","b","mu_input","mu_original","mu_initial",
                           "teacher32targets","mse","task_start_mse","derivative_absmean",
                           "activeunit_frac",optional=("alpha","VEMA"))
            w0,b0,w1,b1 = (np.asarray(z,dtype=np.float64) for z in
                           (previous["W"],previous["b"],current["W"],current["b"]))
        else:
            current = _npz(current_path,"W1","b1",optional=("adaptive_alpha1","adaptive_alpha2"))
            w0,b0,w1,b1 = (np.asarray(z,dtype=np.float64) for z in
                           (previous["W1"],previous["b1"],current["W1"],current["b1"]))
        if not all(np.all(np.isfinite(z)) for z in (w0,b0,w1,b1)):
            raise ValueError(f"{current_path}: nonfinite first-layer state")
        xp,xc,old_input_source,new_input_source = _inputs(s,initial,task,current,x_previous,bank)
        pcur0=(previous_current if task>1 and s.cell.startswith("X0") else w0@xc.T)
        pcur1=w1@xc.T
        if task==1:
            pold0=pcur0
        else:
            pold0=previous_current
        if task==1:
            pref0=w0@xref.T
        else:
            pref0=previous_reference
        pref1=pcur1 if s.cell.startswith("X0") else w1@xref.T
        reference=_projected_terms(pref0,pref1)
        actual_current=_projected_terms(pcur0,pcur1)
        old_v=_projected_terms(pold0,pold0)["V"]
        g=actual_current["V"]-old_v
        actual={**actual_current,"V":old_v,"Vprev_current":actual_current["V"],"G_pre":g,
                "identity_residual":actual_current["V_next"]-old_v-actual_current["Q"]-2*actual_current["X"]-g}
        raw=_raw_terms(w0,w1)
        paired=_paired_from_a_e(pcur1-pcur0+(b1-b0)[:,None],pcur0-pold0)
        if not np.allclose(paired["Q_current"],actual["Q"],rtol=1e-8,atol=1e-8):
            raise ValueError(f"task {task}: paired Q disagrees with current covariance")
        base={"environment":s.environment,"arm":s.arm,"activation":s.activation,"cell":s.cell,
              "seed":s.seed,"task":task,"source_prev":str(_snapshot(s,task-1)),
              "source_next":str(current_path),"input_prev_source":old_input_source,
              "input_current_source":new_input_source,
              "reference_input_source":str(_snapshot(s,0) if bank is None else s.path/"input_bank.npz"),
              "covariance_convention":"population_centered_current_for_Q_X_Gpre",
              "paired_support_size":len(xc)}
        if s.environment=="conda":
            base.update({k:float(current[k]) for k in ("mse","task_start_mse","derivative_absmean","activeunit_frac")})
            base["task_mse_gain"]=base["task_start_mse"]-base["mse"]
            base["mu_prev_source"]=old_input_source
            base["mu_current_source"]=new_input_source
            base["target32_source"]=str(current_path)
            old_target=(initial if task==1 else previous)["teacher32targets"]
            target_delta=current["teacher32targets"].astype(np.float64)-np.asarray(old_target,dtype=np.float64)
            base["target_change_energy"]=float(np.mean(target_delta**2))
            base["target_change_fraction"]=float(np.mean(target_delta!=0))
            base["learner_input_mean_shift_sq"]=float(np.sum((xc.mean(axis=0)-xp.mean(axis=0))**2))
            base["original_context_mean_shift_sq"]=float(np.sum((current["mu_original"][:15]-
                          (initial if task==1 else previous)["mu_original"][:15])**2))
            if "alpha" in current:
                alpha=current["alpha"].astype(np.float64)
                base.update(adaptive_alpha_median=float(np.median(alpha)),
                            adaptive_alpha_clip_frac=float(np.mean((alpha<=.05)|(alpha>=3.))))
        else:
            base.update(diagnostics[task])
            base["task_accuracy_gain"]=float(base["train_acc"])-float(base["task_start_acc"])
            current_labels=_npz(s.path/f"task_{task:03d}_input.npz","labels")["labels"]
            previous_labels=(current_labels if task==1 else
                             _npz(s.path/f"task_{task-1:03d}_input.npz","labels")["labels"])
            base["target_change_fraction"]=float(np.mean(current_labels!=previous_labels))
            base["learner_input_mean_shift_sq"]=float(np.sum((xc.mean(axis=0)-xp.mean(axis=0))**2))
        summed_paired={k:float(np.sum(v,dtype=np.float64)) for k,v in paired.items()
                       if k not in ("mean_A","mean_E")}
        for metric,unit in (("reference",reference),("actual",actual),("raw",raw)):
            summed=_sum_terms(unit)
            if metric=="actual":
                summed["Vprev_old"]=summed["V"]
                summed.update(summed_paired)
                summed["G_pre_abs_over_Q"]=abs(summed["G_pre"])/summed["Q"] if summed["Q"]>0 else math.nan
                summed["E_over_A_energy"]=summed["E_energy"]/summed["A_energy"] if summed["A_energy"]>0 else math.nan
                summed["mean_A_over_A_energy"]=summed["mean_A_sq"]/summed["A_energy"] if summed["A_energy"]>0 else math.nan
                summed["mean_E_over_E_energy"]=summed["mean_E_sq"]/summed["E_energy"] if summed["E_energy"]>0 else math.nan
                summed["paired_mean_shift_sq"]=float(np.sum((paired["mean_A"]+paired["mean_E"])**2))
            else:
                summed["G_pre"]=0.
            summed["covariance_term"]=summed["G_pre"]
            summed["delta_V"]=summed["V_next"]-summed["V"]
            summed["residual"]=summed["delta_V"]-summed["Q"]-2*summed["X"]
            if metric=="reference":
                summed["mean_shift_sq"]=float(np.sum(((w1-w0)@xref.mean(axis=0)+(b1-b0))**2))
            elif metric=="actual":
                summed["mean_shift_sq"]=summed["paired_mean_shift_sq"]
            rows.append({**base,"metric":metric,
                         "c_rho_covariance_source":"current" if metric=="actual" else
                         "fixed_reference" if metric=="reference" else "identity",
                         **summed,
                         "max_abs_unit_identity_residual":float(np.max(np.abs(unit["identity_residual"])))})
            for window,(first,last) in windows.items():
                if first<=task<=last:
                    key=(window,metric)
                    if key not in accum:
                        accum[key]={k:np.zeros_like(unit["V"]) for k in ("V","Q","X","D_norm")}
                    for k in accum[key]:
                        accum[key][k]+=unit[k]
        previous,x_previous,previous_current=current,xc,pcur1
        previous_reference=pref1
    for (window,metric),fields in accum.items():
        first,last=windows[window]
        for unit in range(len(fields["V"])):
            v,q,x=(float(fields[k][unit]) for k in ("V","Q","X"))
            unit_rows.append({"environment":s.environment,"arm":s.arm,"activation":s.activation,
                              "cell":s.cell,"seed":s.seed,"window":window,"metric":metric,
                              "unit":unit,"first_task":first,"last_task":last,"sum_V":v,"sum_Q":q,
                              "sum_X":x,"mean_D_norm":float(fields["D_norm"][unit])/(last-first+1),
                              "balance":-2*x/q if q>0 else math.nan})
    return rows,unit_rows


def bootstrap_median_ci(values, draws: int = 5000, seed: int = 924) -> tuple[float,float]:
    a=np.asarray(values,dtype=np.float64)
    a=a[np.isfinite(a)]
    if not len(a):
        return math.nan,math.nan
    rng=np.random.default_rng(seed)
    draw=a[rng.integers(len(a),size=(draws,len(a)))]
    return tuple(float(x) for x in np.quantile(np.median(draw,axis=1),[.025,.975]))


def low_response_onset(table: pd.DataFrame, environment: str) -> int | None:
    """First endpoint in first run of three, selecting schema by environment."""
    ordered=table.sort_values("task")
    if environment=="conda":
        low=(ordered.derivative_absmean.to_numpy(dtype=float)<1e-8)|(
            ordered.activeunit_frac.to_numpy(dtype=float)<.1)
    elif environment=="mnist":
        low=np.zeros(len(ordered),dtype=bool)
        for layer in (1,2):
            low|=ordered[f"derivative_abs_mean_l{layer}"].to_numpy(dtype=float)<1e-8
            low|=ordered[f"active_unit_frac_l{layer}"].to_numpy(dtype=float)<.1
    else:
        raise ValueError(environment)
    for i in range(len(low)-2):
        if np.all(low[i:i+3]):
            return int(ordered.task.iloc[i])
    return None


def _pre_onset(table: pd.DataFrame,onset: int|None) -> dict:
    empty={"active_window_first_task":math.nan,"active_window_last_task":math.nan,
           "active_window_balance":math.nan,"active_window_activity":math.nan,
           "active_window_loglog_slope":math.nan}
    if onset is None:
        return {"active_window_status":"NO_LOW_RESPONSE",**empty}
    eligible=table[table.task<onset].sort_values("task")
    if len(eligible)<20:
        return {"active_window_status":"INSUFFICIENT_ACTIVE_HISTORY",**empty}
    late=eligible.tail(20)
    q=float(late.Q.sum()); v=float(late.V.mean())
    endpoint_v=np.r_[float(late.V.iloc[0]),late.V_next.to_numpy(dtype=float)]
    endpoint_t=np.r_[int(late.task.iloc[0])-1,late.task.to_numpy(dtype=int)]
    slope=(float(np.polyfit(np.log(endpoint_t),np.log(endpoint_v),1)[0])
           if np.all(endpoint_t>0) and np.all(endpoint_v>0) else math.nan)
    return {"active_window_status":"AVAILABLE","active_window_first_task":int(late.task.iloc[0]),
            "active_window_last_task":int(late.task.iloc[-1]),
            "active_window_balance":-2*float(late.X.sum())/q if q>0 else math.nan,
            "active_window_activity":q/v if v>0 else math.nan,
            "active_window_loglog_slope":slope}


def _closure_rows(block: pd.DataFrame, fit: dict, labels: dict) -> list[dict]:
    held=block[block.task>=fit["heldout_first_task"]].sort_values("task")
    value=float(fit["split_V"])
    rows=[]
    for row in held.itertuples():
        value=(1-2*float(fit["gamma"]))*value+float(fit["qbar"])
        rows.append({**labels,"task":int(row.task),"observed_V":float(row.V_next),
                     "model_V":value,"constant_V":float(fit["split_V"]),
                     "calibration_last_task":fit["calibration_last_task"]})
    return rows


def summarize_window(block: pd.DataFrame, *, environment: str, metric: str,
                     window: str, endpoint: int) -> tuple[dict,list[dict]]:
    """Apply old P1--P4 only to reference; A verdicts only to actual."""
    prefix=block[block.task<=endpoint].sort_values("task")
    fixed=metric in ("reference","raw")
    fit=summarize_task_table(prefix,fixed_covariance=fixed)
    if metric!="reference":
        fit.update(P1="NOT_APPLICABLE",P2="NOT_APPLICABLE",P3="NOT_APPLICABLE")
    late=prefix.tail(int(fit["late_n_transitions"]))
    fit["window"]=window
    fit["window_endpoint"]=endpoint
    fit["late_first_task"]=int(late.task.iloc[0])
    fit["late_last_task"]=int(late.task.iloc[-1])
    fit["late_mean_V"]=float(late.V.mean())
    fit["late_mean_sigma"]=float(np.sqrt(late.V).mean())
    fit["late_mean_sqrtQ"]=float(np.sqrt(late.Q).mean())
    for field in ("Q","X","G_pre","A_energy","E_energy","AE_cross","total_energy",
                  "mean_A_sq","mean_E_sq","centered_E_energy"):
        if field in late:
            fit[f"late_sum_{field}"]=float(late[field].sum())
    for field in ("target_change_fraction","target_change_energy","learner_input_mean_shift_sq",
                  "original_context_mean_shift_sq","task_start_mse","mse","task_mse_gain","task_start_acc","train_acc",
                  "task_accuracy_gain","grad_w1_start_norm","grad_w1_end_norm",
                  "grad_w2_start_norm","grad_w2_end_norm","adaptive_alpha_median",
                  "adaptive_alpha_clip_frac","adaptive_alpha_median_l1","adaptive_alpha_median_l2",
                  "adaptive_alpha_clip_frac_l1","adaptive_alpha_clip_frac_l2"):
        if field in late:
            fit[f"late_{field}"]=float(pd.to_numeric(late[field],errors="coerce").mean())
    if metric=="actual":
        qsum=float(late.Q.sum())
        xsum=float(late.X.sum())
        gsum=float(late.G_pre.sum())
        activity=qsum/float(late.V.mean()) if late.V.mean()>0 else math.nan
        balance=-(2*xsum+gsum)/qsum if qsum>0 else math.nan
        fit["late_A_balance"]=balance
        fit["late_A_activity"]=activity
        fit["late_update_only_balance"]=-2*xsum/qsum if qsum>0 else math.nan
        fit["late_G_pre_over_Q"]=gsum/qsum if qsum>0 else math.nan
        fit["late_abs_G_pre_over_Q"]=float(late.G_pre.abs().sum())/qsum if qsum>0 else math.nan
        vcur=late.Vprev_current.to_numpy(dtype=float)
        qs=late.Q.to_numpy(dtype=float)
        xs=late.X.to_numpy(dtype=float)
        c=np.divide(xs,np.sqrt(vcur*qs),out=np.full(len(late),np.nan),where=(vcur*qs)>0)
        fit["late_update_negative_c_fraction"]=float(np.mean(c<0))
        valid=bool(np.isfinite(balance) and np.isfinite(activity) and
                   .9<=balance<=1.1 and activity>=.1 and
                   np.all(late.V.to_numpy(dtype=float)>0) and
                   np.all(late.V_next.to_numpy(dtype=float)>0))
        fit["A_BALANCE"]="PASS" if valid else "FAIL"
        fit["A_PLATEAU"]=("PASS" if valid and np.isfinite(fit["late_loglog_slope"])
                          and abs(fit["late_loglog_slope"])<=.2
                          and .9<=fit["late_halves_ratio"]<=1.1 else "FAIL")
    else:
        fit.update(A_BALANCE="NOT_APPLICABLE",A_PLATEAU="NOT_APPLICABLE")
    onset=low_response_onset(prefix,environment)
    fit["low_response_onset_task"]=onset if onset is not None else math.nan
    fit["response_censoring"]="LOW_RESPONSE" if onset is not None else "NO_LOW_RESPONSE"
    fit.update(_pre_onset(prefix,onset))
    if metric=="reference":
        fit["dynamics_label"]=("INVALID_METRIC" if fit["late_invalid_V"] else
                               "STOPPED" if fit["stopped"] else
                               "PLATEAU_WITH_ACTIVITY" if fit["P2"]=="PASS" else
                               "NEAR_BALANCE_WITH_ACTIVITY" if fit["P1"]=="PASS" else
                               "TRANSIENT_GROWTH" if fit["late_loglog_slope"]>.2 else "OTHER_ACTIVE")
    else:
        fit["dynamics_label"]="DIAGNOSTIC_ONLY"
    trajectory=_closure_rows(prefix,fit,{"window":window,"metric":metric}) if metric=="reference" else []
    return fit,trajectory


def summarize_series(transitions: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rows,closures,sources=[],[],[]
    keys=["environment","arm","activation","cell","seed","metric"]
    for key,block in transitions.groupby(keys,sort=True):
        labels=dict(zip(keys,key))
        series=Series(labels["environment"],labels["arm"],labels["activation"],
                      labels["cell"],int(labels["seed"]),int(block.task.max()),Path("."))
        for window,(_,endpoint) in _window_specs(series).items():
            fit,curve=summarize_window(block,environment=series.environment,
                                       metric=labels["metric"],window=window,endpoint=endpoint)
            rows.append({**labels,**fit})
            closures.extend({**labels,**r} for r in curve)
            late=block[(block.task>=fit["late_first_task"])&(block.task<=fit["late_last_task"])]
            sources.append({**labels,"window":window,"calibration_first_task":fit["calibration_first_task"],
                            "calibration_last_task":fit["calibration_last_task"],
                            "heldout_first_task":fit["heldout_first_task"],
                            "heldout_last_task":fit["heldout_last_task"],
                            "late_first_task":fit["late_first_task"],"late_last_task":fit["late_last_task"],
                            "first_source":str(block.source_prev.iloc[0]),
                            "last_source":str(late.source_next.iloc[-1]),
                            "reference_input_source":str(block.reference_input_source.iloc[0]),
                            "first_late_input_source":str(late.input_current_source.iloc[0]),
                            "last_late_input_source":str(late.input_current_source.iloc[-1])})
    summary=pd.DataFrame(rows)
    summary["P4"]="NOT_APPLICABLE"
    summary["raw_minus_reference_slope"]=math.nan
    for idx,row in summary[summary.metric=="reference"].iterrows():
        peer=summary[(summary.environment==row.environment)&(summary.arm==row.arm)&
                     (summary.seed==row.seed)&(summary.window==row.window)&(summary.metric=="raw")]
        if len(peer)!=1:
            raise ValueError("missing unique raw slope peer")
        gap=float(peer.late_loglog_slope.iloc[0]-row.late_loglog_slope)
        summary.at[idx,"raw_minus_reference_slope"]=gap
        summary.at[idx,"P4"]="SEPARATED" if row.P2=="PASS" and gap>=.2 else "NOT_ESTABLISHED"
    return summary,pd.DataFrame(closures),pd.DataFrame(sources)


def group_verdicts(summary: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    questions={"reference":{"P1":"late_nearbalance","P2":"late_loglog_slope",
                             "P3":"heldout_mape","P4":"raw_minus_reference_slope"},
               "actual":{"A_BALANCE":"late_A_balance","A_PLATEAU":"late_loglog_slope"}}
    for (env,arm,window,metric),block in summary.groupby(["environment","arm","window","metric"]):
        for question,column in questions.get(metric,{}).items():
            status=block[question].tolist()
            n_pass=sum(x in ("PASS","SEPARATED") for x in status)
            n_uninformative=sum(x=="UNINFORMATIVE" for x in status)
            values=block[column].to_numpy(dtype=float)
            lo,hi=bootstrap_median_ci(values)
            verdict=("INCOMPLETE" if len(block)!=5 else
                     "SEPARATED" if question=="P4" and n_pass>=4 else
                     "PASS" if n_pass>=4 else
                     "UNINFORMATIVE" if question=="P3" and n_uninformative>=4 else
                     "NOT_ESTABLISHED" if question=="P4" else "FAIL")
            rows.append({"environment":env,"arm":arm,"window":window,"metric":metric,
                         "question":question,"verdict":verdict,"n_seed":len(block),
                         "n_pass":n_pass,"n_uninformative":n_uninformative,
                         "statistic":column,"seed_median":float(np.median(values[np.isfinite(values)]))
                         if np.isfinite(values).any() else math.nan,
                         "ci95_low":lo,"ci95_high":hi})
    return pd.DataFrame(rows)


def group_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Five-seed median and bootstrap CI for every numeric summary field."""
    keys=["environment","arm","window","metric"]
    fields=[name for name in summary.select_dtypes(include=[np.number]).columns
            if name not in ("seed",)]
    rows=[]
    for labels,block in summary.groupby(keys):
        prefix=dict(zip(keys,labels))
        for field in fields:
            values=block[field].to_numpy(dtype=float)
            finite=values[np.isfinite(values)]
            lo,hi=bootstrap_median_ci(finite)
            rows.append({**prefix,"field":field,"n_seed":len(finite),
                         "seed_median":float(np.median(finite)) if len(finite) else math.nan,
                         "ci95_low":lo,"ci95_high":hi})
    return pd.DataFrame(rows)


PAIR_FIELDS=("late_nearbalance","late_activity","late_loglog_slope","late_halves_ratio",
             "late_mean_sqrtQ","late_mean_sigma","late_A_balance","late_A_activity",
             "late_update_only_balance","late_G_pre_over_Q","late_abs_G_pre_over_Q",
             "late_sum_A_energy","late_sum_E_energy","late_sum_AE_cross",
             "late_sum_total_energy","late_sum_mean_A_sq","late_sum_mean_E_sq",
             "late_target_change_fraction","late_target_change_energy",
             "late_learner_input_mean_shift_sq","late_original_context_mean_shift_sq",
             "late_task_start_mse","late_mse","late_task_mse_gain",
             "late_task_start_acc","late_train_acc","late_task_accuracy_gain",
             "late_grad_w1_start_norm","late_grad_w1_end_norm",
             "late_grad_w2_start_norm","late_grad_w2_end_norm",
             "late_adaptive_alpha_median","late_adaptive_alpha_clip_frac",
             "late_adaptive_alpha_median_l1","late_adaptive_alpha_clip_frac_l1",
             "late_adaptive_alpha_median_l2","late_adaptive_alpha_clip_frac_l2")


def _comparisons() -> list[tuple[str,str,str,str,str]]:
    """All registered factorial contrasts, plus paired k and Snake controls."""
    pairs=[]
    for act in ("LR","SNA06"):
        for k in (1,7):
            for y in (0,1):
                pairs.append(("conda",f"{act}_k{k}_X0Y{y}",f"{act}_k{k}_X1Y{y}",
                              "primary400","input_X1_minus_X0_given_Y"))
            for x in (0,1):
                pairs.append(("conda",f"{act}_k{k}_X{x}Y0",f"{act}_k{k}_X{x}Y1",
                              "primary400","target_Y1_minus_Y0_given_X"))
        for cell in CELLS:
            pairs.append(("conda",f"{act}_k1_{cell}",f"{act}_k7_{cell}",
                          "primary400","k7_minus_k1"))
    for k in (1,7):
        for cell in CELLS:
            pairs.append(("conda",f"LR_k{k}_{cell}",f"SNA06_k{k}_{cell}",
                          "primary400","SNA06_minus_LR"))
    # Duplicate-free, and every CondA contrast also has the secondary 100-task window.
    conda=list(dict.fromkeys(pairs))
    pairs+= [(env,a,b,"secondary100",design) for env,a,b,_,design in conda]
    for act in ("LR","SNA06"):
        for y in (0,1):
            pairs.append(("mnist",f"{act}_X0Y{y}",f"{act}_X1Y{y}",
                          "primary50","input_X1_minus_X0_given_Y"))
        for x in (0,1):
            pairs.append(("mnist",f"{act}_X{x}Y0",f"{act}_X{x}Y1",
                          "primary50","target_Y1_minus_Y0_given_X"))
        for cell in CELLS:
            pairs.append(("mnist",f"LR_{cell}",f"SNA06_{cell}",
                          "primary50","SNA06_minus_LR"))
    for act in ("SN05","SNA03"):
        pairs.append(("mnist","LR_X0Y1",f"{act}_X0Y1", "primary50",f"{act}_minus_LR"))
        pairs.append(("mnist","SNA06_X0Y1",f"{act}_X0Y1", "primary50",f"{act}_minus_SNA06"))
    pairs.append(("mnist","SNA03_X0Y1","SN05_X0Y1","primary50","fixed_SN05_minus_adaptive_SNA03"))
    pairs.append(("mnist","LR_X0Y1","SNA06_X0Y1","long150","SNA06_minus_LR"))
    return list(dict.fromkeys(pairs))


def paired_comparisons(summary: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    index={(r.environment,r.arm,r.window,r.metric,int(r.seed)):r for r in summary.itertuples()}
    rows=[]
    specs=[(env,a,b,window,window,design) for env,a,b,window,design in _comparisons()]
    specs += [("conda",arm,arm,"secondary100","primary400","400_minus_100_same_arm")
              for arm in sorted(summary.loc[summary.environment.eq("conda"),"arm"].unique())]
    specs += [("mnist",f"{act}_X0Y1",f"{act}_X0Y1", "primary50","long150",
               "150_minus_50_same_arm") for act in ("LR","SNA06")]
    for env,ref_arm,cmp_arm,ref_window,cmp_window,design in specs:
        window=cmp_window if ref_window==cmp_window else f"{cmp_window}_vs_{ref_window}"
        for metric in ("raw","reference","actual"):
            for seed in SEEDS:
                ref=index.get((env,ref_arm,ref_window,metric,seed))
                cmp=index.get((env,cmp_arm,cmp_window,metric,seed))
                row={"environment":env,"reference_arm":ref_arm,"comparison_arm":cmp_arm,
                     "window":window,"reference_window":ref_window,"comparison_window":cmp_window,
                     "design":design,"metric":metric,"seed":seed,
                     "status":"COMPLETE" if ref is not None and cmp is not None else "INCOMPLETE"}
                for field in PAIR_FIELDS:
                    for prefix in ("reference","comparison","difference","ratio"):
                        row[f"{prefix}_{field}"]=math.nan
                if ref is not None and cmp is not None:
                    for field in PAIR_FIELDS:
                        a,b=getattr(ref,field,math.nan),getattr(cmp,field,math.nan)
                        a=float(a) if a is not None else math.nan
                        b=float(b) if b is not None else math.nan
                        row[f"reference_{field}"]=a
                        row[f"comparison_{field}"]=b
                        row[f"difference_{field}"]=b-a if np.isfinite(a) and np.isfinite(b) else math.nan
                        row[f"ratio_{field}"]=b/a if np.isfinite(a) and a!=0 and np.isfinite(b) else math.nan
                rows.append(row)
    pairs=pd.DataFrame(rows)
    group=[]
    group_keys=["environment","reference_arm","comparison_arm","window",
                "reference_window","comparison_window","design","metric"]
    for keys,block in pairs.groupby(group_keys):
        labels=dict(zip(group_keys,keys))
        for field in PAIR_FIELDS:
            for kind in ("difference","ratio"):
                column=f"{kind}_{field}"
                values=pd.to_numeric(block[column],errors="coerce").to_numpy(dtype=float)
                finite=values[np.isfinite(values)]
                lo,hi=bootstrap_median_ci(finite)
                group.append({**labels,"field":field,"contrast":kind,"n_seed":len(finite),
                              "seed_median":float(np.median(finite)) if len(finite) else math.nan,
                              "ci95_low":lo,"ci95_high":hi})
    return pairs,pd.DataFrame(group)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(none)"
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]
    for row in df.reindex(columns=columns).itertuples(index=False,name=None):
        lines.append("| "+" | ".join(f"{v:.4g}" if isinstance(v,(float,np.floating)) and np.isfinite(v)
                                     else str(v) for v in row)+" |")
    return "\n".join(lines)


def write_summary(path: Path, completion: pd.DataFrame, summary: pd.DataFrame,
                  verdict: pd.DataFrame, paired_group: pd.DataFrame, *, complete: bool,
                  scope_label: str = "all registered arms") -> None:
    lines=["# Input-scope effective-displacement validation", "",
           f"Run state: **{'COMPLETE' if complete else 'INTERIM / INCOMPLETE'}**; scope: **{scope_label}**. {int(completion.status.eq('COMPLETE').sum())}/{len(completion)} selected series complete. Full registration has 130 series.", "",
           "## Interpretation and definitions", "",
           "Reference P1–P4 use the first fixed input bank and the previous registered rules. Actual A_BALANCE/A_PLATEAU use the current bank for Q and X, with G_pre measured before the weight update. In actual rows V=Vprev_old for the ledger and trends, while c and rho use Vprev_current so their geometry matches Q/X; the source is explicit in `c_rho_covariance_source`. They are separate questions. An exact variance or paired-bank identity is a bookkeeping check, not scientific support.", "",
           "When this run is INCOMPLETE, displayed group verdicts are provisional diagnostics, not final claims.", "",
           "X0Y1 removes the persistent task cue while changing targets; X1−X0 is a paired context contrast, not a pure causal effect of input mean. Fixed input does not guarantee stationary updates or a plateau. STOPPED means low relative update supply, not necessarily zero gradient or inability to learn. LOW_RESPONSE is an operational three-task marker, not proven loss of plasticity.", "",
           "CondA uses changing mean/support with constant centered covariance; MNIST X1 also changes centered covariance. The mean channel and paired-ID function change are reported separately from centered V/Q/X. A_BALANCE may pass through input-switch cancellation and is not by itself evidence of update-driven pruning. No observed G_pre is used as held-out prediction.", "",
           "## Fixed-input sufficiency on observed horizons", ""]
    fixed=verdict[(verdict.metric=="reference")&(verdict.question.isin(["P2","P4"]))&
                  (((verdict.environment=="mnist")&(verdict.arm.isin(["LR_X0Y1","SNA06_X0Y1"]))&
                    verdict.window.isin(["primary50","long150"]))|
                   ((verdict.environment=="conda")&verdict.arm.str.contains("_X0Y1")&
                    verdict.window.eq("primary400")))]
    lines += [_markdown_table(fixed,["environment","arm","window","question","verdict","n_pass","seed_median","ci95_low","ci95_high"]),"",
              "A finite-horizon failure supports only insufficiency under the observed optimizer, activation, and horizon.","",
              "## All group verdicts", "",
              _markdown_table(verdict,["environment","arm","window","metric","question","verdict","n_pass","n_seed","seed_median","ci95_low","ci95_high"]),"",
              "## Paired comparisons", "",
              "All registered X/Y, k, activation, Snake, and horizon contrasts appear seed by seed in `paired_seed.csv`, with 5-seed median and bootstrap95% intervals in `paired_group.csv`. `group_stats.csv` gives the same intervals for every numeric seed summary field. Every listed field is reported, including null contrasts; no outcome field was selected after viewing results.","",
              "## Traceability", "",
              "`source_paths.csv` records snapshot and input paths, `source_windows.csv` records fit/held-out/late windows, and `spec_hashes.csv` records the exact preregistration hashes. `transitions.csv.gz` contains per-task raw/reference/actual values and the paired decomposition; `transitions_compression.json` records its round-trip SHA256 check. `unit_late.csv` holds late per-unit aggregates. `closure.csv` contains fixed-reference held-out trajectories only.",""]
    if not summary.empty:
        relevant=summary[summary.metric=="reference"]
        lines += ["## Response and learning context", "",
                  _markdown_table(relevant,["environment","arm","window","seed","response_censoring",
                                            "low_response_onset_task","active_window_status",
                                            "dynamics_label","late_task_start_mse","late_mse",
                                            "late_task_start_acc","late_train_acc",
                                            "late_grad_w1_start_norm","late_grad_w1_end_norm"]),""]
    path.write_text("\n".join(lines))


def run_analysis(raw_root: Path, out: Path, *, allow_incomplete: bool=False,
                 make_plots: bool=True, activation_filter: str|None=None) -> dict[str,pd.DataFrame]:
    raw_root,out=Path(raw_root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    series=expected_series(raw_root)
    if activation_filter is not None:
        if activation_filter != "LR":
            raise ValueError("only registered LR interim filtering is supported")
        series=[s for s in series if s.activation==activation_filter]
    scope_label=("LR-only interim, 60/130 registered series selected" if activation_filter=="LR"
                 else "all 130 registered series")
    pd.DataFrame([{"scope":scope_label,"registered_series":130,"selected_series":len(series),
                   "activation_filter":activation_filter or "ALL",
                   "final_full_analysis":activation_filter is None}]).to_csv(out/"scope.csv",index=False)
    completion=pd.DataFrame([preflight(s) for s in series])
    completion.to_csv(out/"completion.csv",index=False)
    if (completion.status!="COMPLETE").any() and not allow_incomplete:
        raise RuntimeError(f"{(completion.status!='COMPLETE').sum()} of {len(completion)} seed series incomplete; see completion.csv")
    transitions,units=[],[]
    for s,ok in zip(series,completion.status=="COMPLETE"):
        if not ok:
            continue
        try:
            task_rows,unit_rows=transition_rows(s)
            transitions.extend(task_rows)
            units.extend(unit_rows)
        except Exception as exc:
            mask=(completion.environment==s.environment)&(completion.arm==s.arm)&(completion.seed==s.seed)
            completion.loc[mask,["status","reason"]]=["ANALYSIS_FAILED",repr(exc)]
            completion.to_csv(out/"completion.csv",index=False)
            if not allow_incomplete:
                raise RuntimeError(f"analysis failed for {s.environment}/{s.arm}/seed{s.seed}: {exc}") from exc
    tdf,udf=pd.DataFrame(transitions),pd.DataFrame(units)
    compression=_write_gzip_csv(tdf,out/"transitions.csv.gz")
    (out/"transitions_compression.json").write_text(json.dumps(compression,indent=2,sort_keys=True)+"\n")
    udf.to_csv(out/"unit_late.csv",index=False)
    source_paths=completion.copy()
    source_paths["snapshot_initial"]=[str(_snapshot(s,0)) for s in series]
    source_paths["snapshot_final"]=[str(_snapshot(s,s.tasks)) for s in series]
    source_paths["input_bank"]=[str(s.path/"input_bank.npz") if s.environment=="mnist"
                                else str(_snapshot(s,0)) for s in series]
    source_paths.to_csv(out/"source_paths.csv",index=False)
    pd.DataFrame([{"path":name,"sha256":value} for name,value in _spec_hashes().items()]).to_csv(
        out/"spec_hashes.csv",index=False)
    if tdf.empty:
        empty=pd.DataFrame()
        for name in ("seed_summary","source_windows","closure","verdict","group_stats","paired_seed","paired_group"):
            empty.to_csv(out/f"{name}.csv",index=False)
        write_summary(out/"summary.md",completion,empty,empty,empty,complete=False,
                      scope_label=scope_label)
        return {"completion":completion,"transitions":empty}
    summary,closure,sources=summarize_series(tdf)
    verdict=group_verdicts(summary)
    group_stats=group_statistics(summary)
    pairs,paired_group=paired_comparisons(summary)
    outputs={"completion":completion,"transitions":tdf,"unit_late":udf,
             "seed_summary":summary,"source_windows":sources,"closure":closure,
             "verdict":verdict,"group_stats":group_stats,"paired_seed":pairs,"paired_group":paired_group}
    for name,table in outputs.items():
        if name not in ("completion","transitions","unit_late"):
            table.to_csv(out/f"{name}.csv",index=False)
    complete=bool(completion.status.eq("COMPLETE").all() and activation_filter is None)
    write_summary(out/"summary.md",completion,summary,verdict,paired_group,
                  complete=complete,scope_label=scope_label)
    if make_plots:
        from analysis.effdisp_inputscope_0924.plots import make_plots as render
        render(tdf,summary,closure,paired_group,out)
    return outputs


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root",type=Path,default=DEFAULT_RAW)
    parser.add_argument("--out",type=Path,default=REPO/"results"/"effdisp_inputscope_0924")
    parser.add_argument("--allow-incomplete",action="store_true",help="diagnostic output; never mark as final")
    parser.add_argument("--activation",choices=("LR",),help="registered LR-only interim scope")
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    run_analysis(args.raw_root,args.out,allow_incomplete=args.allow_incomplete,
                 make_plots=not args.no_plots,activation_filter=args.activation)


if __name__=="__main__":
    main()

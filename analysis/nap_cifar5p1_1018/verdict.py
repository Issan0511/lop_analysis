#!/usr/bin/env python3
"""Exact seed-paired NaP comparisons. Never join rows by input order."""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from src import nap_cifar5p1_1018 as N

WINDOW = (21, 23, 25, 27, 29)


def exact_sign(d):
    d = np.asarray(d, dtype=np.float64)
    if not np.isfinite(d).all():
        raise ValueError("nonfinite paired difference")
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    n = npos + nneg
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(npos, nneg)+1)) / 2**n) if n else 1.0
    return dict(p=p, n_nonzero=n, positive=npos, negative=nneg, ties=len(d)-n)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda j:(pvalues[j], j))
    adjusted, running = [0.0]*len(pvalues), 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (len(pvalues)-rank)*pvalues[j]))
        adjusted[j] = running
    return adjusted


def classify(d, corrected_p):
    d = np.asarray(d, dtype=np.float64)
    if d.shape != (10,) or not np.isfinite(d).all():
        raise ValueError("ten finite paired seed differences required")
    median = float(np.median(d))
    interval = np.sort(d)[[1,8]].tolist()
    if corrected_p < .05 and median > 0:
        label = "NAP_WINS"
    elif corrected_p < .05 and median < 0:
        label = "COMPARATOR_WINS"
    elif corrected_p >= .05 and interval[0] >= -.005 and interval[1] <= .005:
        label = "EQUIVALENT_WITHIN_0.005"
    else:
        label = "UNRESOLVED"
    return dict(label=label, paired_mean=float(d.mean()), paired_median=median,
                differences_by_seed={str(s):float(v) for s,v in enumerate(d)},
                median_interval=interval, median_interval_coverage=.978515625,
                holm_p=corrected_p, **exact_sign(d))


def load_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def seed_window(rows, seeds):
    cells = {}
    for row in rows:
        key = (int(row["seed"]), int(row["task"]))
        if key in cells:
            raise ValueError(f"duplicate seed/task {key}")
        cells[key] = row
    expected = {(s,t) for s in seeds for t in range(1,31)}
    if set(cells) != expected:
        raise ValueError("INCOMPLETE: exact ten seeds x thirty unique tasks required")
    values = {}
    for s in seeds:
        v=[]
        for t in range(1,31):
            q=cells[s,t]
            if int(q["hard"]) != int(t%2 == 1) or q.get("status", "COMPLETE") != "COMPLETE":
                raise ValueError("invalid task parity or failed seed")
            x=float(q["online_acc"])
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError("invalid online accuracy")
            if t in WINDOW:
                v.append(x)
        values[s]=float(np.mean(v))
    return values


def compare(nap, comparators):
    if set(comparators) != {"SNA", "KKT1"} or set(nap) != set(range(10)):
        raise ValueError("exact registered primary family required")
    names = ("KKT1", "SNA")
    differences=[]
    for name in names:
        if set(comparators[name]) != set(range(10)):
            raise ValueError("incorrect comparator seeds")
        differences.append(np.asarray([nap[s]-comparators[name][s] for s in range(10)]))
    adjusted=holm([exact_sign(d)["p"] for d in differences])
    return {name:classify(d,p) for name,d,p in zip(names,differences,adjusted)}


def committed_artifact(path):
    """Bind legacy output bytes to the immutable main-history result commit."""
    name=str(Path(path).resolve().relative_to(N.ROOT))
    commit=subprocess.check_output(["git","log","-1","--format=%H","--",name],
                                    cwd=N.ROOT,text=True).strip()
    if not commit or subprocess.run(["git","merge-base","--is-ancestor",commit,"main"],
                                    cwd=N.ROOT,capture_output=True).returncode:
        raise ValueError("legacy artifact is not committed on main")
    historical=subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=N.ROOT)
    digest=__import__("hashlib").sha256(historical).hexdigest()
    if digest!=N.sha_file(path):
        raise ValueError("legacy artifact differs from immutable result commit")
    return dict(path=name,commit=commit,sha256=digest)


def qualification(arm, directory, audit):
    """Mechanical reuse proof; historical changes need an independent equivalence audit.

    A performance mean alone never grants reuse. The historical source identity is
    checked using the recorded git commit, not the checkout that happens to be open.
    """
    directory=directory.resolve()
    rows_path, prov_path=directory/"per_task.csv", directory/"provenance.json"
    errors=N.check_audit(audit)
    p=json.loads(prov_path.read_text())
    committed={}
    for path in (rows_path,prov_path):
        try:
            committed[path.name]=committed_artifact(path)
        except (ValueError,subprocess.CalledProcessError) as e:
            errors.append("LEGACY_OUTPUT_UNBOUND:"+str(e))
    verified_data=None
    try:
        verified_data=N.C.Cifar100(N.DEFAULT_DATA).sha256
        if p.get("data_sha256") != verified_data:
            errors.append("DATASET_DIFFERS_FROM_VERIFIED_CIFAR100")
    except (OSError,ValueError) as e:
        errors.append("CIFAR100_VERIFICATION_FAILED:"+str(e))
    required=dict(arm=arm, cond="std", seeds=list(range(10)), R=10, lr=1e-4,
                  n_tasks=30, steps_per_task=780, batch=32, dims=[3072,100,100,100],
                  n_classes=100, hard_classes=5, optimizer="adam", weight_decay=0.0)
    for key,value in required.items():
        if p.get(key) != value:
            errors.append("METADATA_"+key)
    if p.get("divergences"):
        errors.append("DIVERGENCES")
    if p.get("iv_kind", "none") != "none" or p.get("intervention", "none") != "none":
        errors.append("INTERVENTION")
    if p.get("std",{}).get("mean") != list(N.C.STD_MEAN) or p.get("std",{}).get("std") != list(N.C.STD_STD):
        errors.append("CIFAR100_STD")
    for s in range(10):
        correct=N.sha_tensor(__import__("torch").tensor(N.C.seed_classes(s), dtype=__import__("torch").int64))
        if p.get("class_sha256",{}).get(str(s)) != correct:
            errors.append(f"CLASS_STREAM_{s}")
    try:
        records=load_rows(rows_path)
        values=seed_window(records, list(range(10)))
        if any(int(r["slot"]) != int(r["seed"]) for r in records):
            errors.append("SEED_SLOT_ORDER")
    except (ValueError, KeyError) as e:
        values=None; errors.append(str(e))
    commit=p.get("git_hash", "")
    historical={}
    for name in N.HOST_SOURCES:
        old=subprocess.run(["git", "show", f"{commit}:{name}"], cwd=N.ROOT, capture_output=True)
        if old.returncode:
            errors.append("HISTORICAL_SOURCE_MISSING:"+name)
        else:
            historical[name]=__import__("hashlib").sha256(old.stdout).hexdigest()
            if historical[name] != N.sha_file(N.ROOT/name):
                errors.append("HISTORICAL_EQUIVALENCE_AUDIT_REQUIRED:"+name)
    return dict(arm=arm, all_pass=not errors, eligible=not errors, errors=errors,
                path=str(directory), per_task_sha256=N.sha_file(rows_path),
                provenance_sha256=N.sha_file(prov_path), original_git_hash=commit,
                data_sha256=p.get("data_sha256"),
                verified_data_sha256=verified_data,committed_artifacts=committed,
                current_source_sha256={n:N.sha_file(N.ROOT/n) for n in N.HOST_SOURCES},
                historical_source_sha256=historical,
                integration_audit_sha256=N.sha_file(audit) if audit.is_file() else None,
                qualified_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                seed_window=values)


def validate_qualification(manifest):
    q=json.loads(Path(manifest).read_text())
    directory=Path(q["path"])
    if q.get("eligible") is not True or q.get("all_pass") is not True:
        raise ValueError("comparator reuse unqualified")
    for name, key in (("per_task.csv", "per_task_sha256"), ("provenance.json", "provenance_sha256")):
        if N.sha_file(directory/name) != q.get(key):
            raise ValueError("reused comparator changed since qualification")
        proof=q.get("committed_artifacts",{}).get(name)
        if not proof or committed_artifact(directory/name)!=proof:
            raise ValueError("reused comparator lacks immutable historical artifact proof")
    if q.get("data_sha256") != q.get("verified_data_sha256"):
        raise ValueError("reused comparator dataset is unverified")
    if q.get("current_source_sha256") != {n:N.sha_file(N.ROOT/n) for n in N.HOST_SOURCES}:
        raise ValueError("host sources changed since qualification")
    return q, seed_window(load_rows(directory/"per_task.csv"), list(range(10)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd", required=True)
    q=sub.add_parser("qualify")
    q.add_argument("--arm", choices=("R","SNA","KKT1"), required=True)
    q.add_argument("--directory", type=Path, required=True)
    q.add_argument("--integration-audit", type=Path, default=N.DEFAULT_AUDIT)
    q.add_argument("--out", type=Path, required=True)
    v=sub.add_parser("verdict")
    v.add_argument("--nap", type=Path, default=N.OUT/"eval/NaP")
    for arm in ("R", "SNA", "KKT1"):
        v.add_argument("--"+arm.lower()+"-qualification", type=Path, required=True)
    v.add_argument("--out", type=Path, default=N.OUT/"verdict.json")
    a=ap.parse_args()
    if a.cmd=="qualify":
        q=qualification(a.arm, a.directory, a.integration_audit)
        N.write_json(a.out, q)
        raise SystemExit(0 if q["eligible"] else 2)
    prov=json.loads((a.nap/"provenance.json").read_text())
    if prov.get("status") != "COMPLETE" or prov.get("phase") != "eval" or prov.get("source_sha256") != N.source_hashes():
        raise SystemExit("NaP evaluation incomplete or source mismatch")
    for filename,digest in prov.get("outputs_sha256",{}).items():
        if N.sha_file(a.nap/filename) != digest:
            raise SystemExit("NaP output changed after completion")
    if "per_task.csv" not in prov.get("outputs_sha256",{}):
        raise SystemExit("NaP per-task artifact not fixed")
    nap=seed_window(load_rows(a.nap/"per_task.csv"), list(range(10)))
    comparators, manifests={},{}
    for arm in ("R","SNA","KKT1"):
        manifest, values=validate_qualification(getattr(a,arm.lower()+"_qualification"))
        if manifest["arm"] != arm:
            raise ValueError("wrong comparator qualification")
        if manifest.get("data_sha256") != prov.get("data_sha256"):
            raise ValueError("comparator dataset differs from NaP")
        comparators[arm]=values; manifests[arm]=manifest
    result=dict(status="COMPLETE", primary=compare(nap, {k:comparators[k] for k in ("SNA","KKT1")}),
                nap_window=nap, R_difference_report_only={s:nap[s]-comparators["R"][s] for s in nap},
                comparator_qualifications=manifests, prediction="SNA/KKT1 OPEN_COMPARISON",
                window=list(WINDOW), family="2 two-sided exact sign tests; Holm alpha=.05",
                simultaneous_median_coverage_lower_bound=.95703125)
    N.write_json(a.out, result)


if __name__=="__main__":
    main()

"""Strict, seed-paired analysis of elu_specificity_1112; never runs a model.

All estimates use saved full-training-set measurements. A bootstrap draw selects
one whole seed cluster, with the same draws reused across arms, ages and LN.
Missing T90 values are right censored; they are never replaced by the horizon.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("S1", "S01", "P1", "P01", "Z_keep", "Z_clear", "F0")
ACTS_A = ("ELU", "R", "LR", "SILU")
STUDIES = {"A": dict(model="mlp", seeds=tuple(range(10)), acts=ACTS_A,
                     ages=(1, 10), horizon=4000, n_examples=1200, every=100),
           "B": dict(model="vit", seeds=tuple(range(10252010, 10252015)),
                     acts=("R", "ELU"), ages=(30,), horizon=1500,
                     n_examples=2500, every=25)}
BOOT = 10000
METRICS = ("U", "U_acute", "U_late", "ce_area", "ce_auc_mean",
           "accuracy_end", "ce_end", "T90_first", "T90_sustained")
T90 = ("T90_first", "T90_sustained")
FACTORS = ("study", "model", "activation", "ln", "t_end", "arm")
BASE = {
    "B_S": {"S1": 1, "S01": -1},
    "B_P": {"P1": 1, "P01": -1},
    "D_1": {"S1": 1, "P1": -1},
    "D_01": {"S01": 1, "P01": -1},
    "scale_by_P": {"S1": 1, "S01": -1, "P1": -1, "P01": 1},
    "Z_S": {"S1": 1, "Z_keep": -1},
    "Z_P": {"P1": 1, "Z_keep": -1},
    "M_Z": {"Z_keep": 1, "Z_clear": -1},
    "body_utility": {"Z_clear": 1, "F0": -1},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def finite_tree(value, path="record"):
    if isinstance(value, dict):
        for k, v in value.items():
            finite_tree(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            finite_tree(v, f"{path}[{i}]")
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(value), f"Nonfinite number: {path}")


def no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def expected_grid(study):
    info = STUDIES[study]
    return sorted({0, 1, 5, 25, 50, 100, 199, 200, 201, info["horizon"],
                   *range(0, info["horizon"] + 1, info["every"])})


def expected_keys():
    return {(st, act, ln, seed, age, arm)
            for st, info in STUDIES.items()
            for act, ln, seed, age, arm in itertools.product(
                info["acts"], ("off", "on"), info["seeds"], info["ages"], ARMS)}


def record_key(r):
    return tuple(r[k] for k in ("study", "activation", "ln", "seed", "t_end", "arm"))


def validate_record(r):
    require(isinstance(r, dict), "Curve JSON must contain an object")
    required = {"study", "model", "activation", "source_activation", "ln", "seed",
                "t_end", "arm", "horizon", "n_examples", "complete", "source_id",
                "source_sha256", "provenance", "checks", "tracks", "curve"}
    require(required <= r.keys(), f"Missing fields: {sorted(required-r.keys())}")
    finite_tree(r)
    require(r["study"] in STUDIES, "Unknown study")
    cfg = STUDIES[r["study"]]
    require(r["model"] == cfg["model"], "Study/model mismatch")
    for field in ("seed", "t_end", "horizon", "n_examples"):
        require(type(r[field]) is int, f"{field} must be an integer, not bool/float")
    require(r["seed"] in cfg["seeds"], "Wrong seed")
    require(r["t_end"] in cfg["ages"], "Wrong task end")
    require(r["activation"] in cfg["acts"], "Wrong activation")
    require(r["source_activation"] == (r["activation"] if r["study"] == "A" else "R"),
            "Wrong source activation")
    require(r["ln"] in ("off", "on"), "Wrong LN level")
    require(r["arm"] in ARMS, "Wrong arm")
    require(r["complete"] is True, "Incomplete curve")
    require(r["horizon"] == cfg["horizon"], "Wrong horizon (smoke is not main)")
    require(r["n_examples"] == cfg["n_examples"], "Wrong fixed-set example count")
    if "steps_run" in r:
        require(type(r["steps_run"]) is int and r["steps_run"] == cfg["horizon"],
                "Incomplete/wrong steps_run")
    require(isinstance(r["source_id"], str) and bool(r["source_id"]), "Missing source ID")
    require(isinstance(r["source_sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", r["source_sha256"]) is not None,
            "Invalid source SHA256")
    for field in ("provenance", "checks"):
        require(isinstance(r[field], dict) and bool(r[field]), f"Empty {field}")
    require(isinstance(r["tracks"], (list, dict)), "Invalid tracks")
    if "passed" in r["checks"]:
        require(r["checks"]["passed"] is True, "Runner checks did not pass")
    curve = r["curve"]
    require(isinstance(curve, list) and bool(curve), "Empty curve")
    for q in curve:
        require(isinstance(q, dict) and {"step", "acc", "ce"} <= q.keys(),
                "Missing curve step/acc/ce")
        require(type(q["step"]) is int, "Curve step must be an integer")
        for metric in ("acc", "ce"):
            require(isinstance(q[metric], (int, float)) and not isinstance(q[metric], bool),
                    f"Invalid {metric}")
        require(0 <= q["acc"] <= 1, "Accuracy outside [0,1]")
        require(q["ce"] >= 0, "Negative CE")
    steps = [q["step"] for q in curve]
    require(steps == expected_grid(r["study"]), "Missing, duplicate, unordered or extra evaluation step")
    return (np.asarray(steps, dtype=float),
            np.asarray([q["acc"] for q in curve], dtype=float),
            np.asarray([q["ce"] for q in curve], dtype=float))


def load_records(raw):
    paths = sorted(Path(raw).rglob("curve_*.json"))
    require(bool(paths), f"No curve_*.json files below {raw}")
    records = []
    for path in paths:
        try:
            r = json.loads(path.read_text(), object_pairs_hook=no_duplicate_keys,
                           parse_constant=reject_constant)
            validate_record(r)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}: {error}") from error
        r["_path"] = str(path.resolve())
        r["_sha256"] = sha256(path)
        records.append(r)
    return records


def validate_design(records):
    actual = []
    source_groups = defaultdict(set)
    source_ids = {}
    execution_hashes, spec_hashes, code_hashes = set(), set(), set()
    for r in records:
        validate_record(r)
        actual.append(record_key(r))
        group = (r["study"], r["activation"] if r["study"] == "A" else "R",
                 r["ln"], r["seed"], r["t_end"])
        source_groups[group].add((r["source_id"], r["source_sha256"]))
        prior = source_ids.setdefault(r["source_id"], group)
        require(prior == group, f"Source ID reused across distinct source conditions: {r['source_id']}")
        for field, values in (("execution_git_hash", execution_hashes),
                              ("spec_sha256", spec_hashes), ("code_sha256", code_hashes)):
            v = r["provenance"].get(field)
            require(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}" if field == "execution_git_hash"
                                                     else r"[0-9a-f]{64}", v) is not None,
                    f"Missing/invalid provenance {field}")
            values.add(v)
    require(len(actual) == len(set(actual)), "Duplicate curve design key")
    expected, seen = expected_keys(), set(actual)
    require(seen == expected,
            f"Incomplete/wrong design: expected {len(expected)}, got {len(seen)}; "
            f"missing={sorted(expected-seen)[:12]}, extra={sorted(seen-expected)[:12]}")
    require(all(len(v) == 1 for v in source_groups.values()),
            "Different source IDs/hashes across paired arms/current activations")
    # Different implementation/spec hashes require a deliberate complete rerun;
    # combining outputs from an incomplete failed attempt is not permitted.
    require(len(execution_hashes) == 1, "Mixed execution commits")
    require(len(spec_hashes) == 1, "Mixed specification hashes")
    require(len(code_hashes) == 1, "Mixed execution code hashes")
    return dict(complete=True, expected=1260, actual=len(records),
                counts={st: sum(r["study"] == st for r in records) for st in STUDIES},
                sources=len(source_groups), execution_git_hash=next(iter(execution_hashes)),
                spec_sha256=next(iter(spec_hashes)), code_sha256=next(iter(code_hashes)),
                source_hash_consistency=True,
                source_file_bytes_verified=False,
                source_verification_note="This analysis checks recorded source identity consistency. "
                    "Source-file byte hashes and archive relocation require the separate provenance audit.")


def numeric_leaves(value, prefix):
    """Flatten numeric diagnostics without silently reinterpreting their units."""
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(numeric_leaves(child, f"{prefix}_{key}"))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.update(numeric_leaves(child, f"{prefix}_{i}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value
    return out


def metric_row(r):
    steps, acc, ce = validate_record(r)
    h = r["horizon"]
    # Canonical MLP accuracy is serialized from float32; 1080/1200 is
    # 0.8999999761581421. The next attainable fraction is far below this boundary.
    threshold = float(np.float32(.9)) if r["study"] == "A" else .9
    hit = acc >= threshold
    first = np.flatnonzero(hit)
    sustained = np.flatnonzero(np.logical_and.accumulate(hit[::-1])[::-1])
    early, late = steps <= 200, steps >= 200
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    row = {k: r[k] for k in (*FACTORS, "seed", "source_activation", "source_id",
                            "source_sha256", "horizon", "n_examples")}
    row.update(U=float(integrate(1 - acc, steps) / h),
               U_acute=float(integrate(1 - acc[early], steps[early]) / 200),
               U_late=float(integrate(1 - acc[late], steps[late]) / (h - 200)),
               ce_area=float(integrate(ce, steps)),
               ce_auc_mean=float(integrate(ce, steps) / h),
               accuracy_end=float(acc[-1]), ce_end=float(ce[-1]),
               T90_first=int(steps[first[0]]) if len(first) else None,
               T90_sustained=int(steps[sustained[0]]) if len(sustained) else None,
               T90_first_censored=not bool(len(first)),
               T90_sustained_censored=not bool(len(sustained)),
               T90_sustained_observation_duration=h - int(steps[sustained[0]]) if len(sustained) else None,
               censor_horizon=h, T90_threshold=threshold,
               initial_accuracy=float(acc[0]), initial_ce=float(ce[0]),
               raw_path=r.get("_path", ""), raw_sha256=r.get("_sha256", ""))
    for point in (0, 25, 200, h):
        q = next(q for q in r["curve"] if q["step"] == point)
        row.update(numeric_leaves({k: v for k, v in q.items() if k != "step"}, f"step{point}"))
    row.update(numeric_leaves(r["tracks"], "track"))
    return row


def bootstrap_indices(study):
    key = f"elu_specificity_1112/common_seed_clusters/{study}/v1"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    n = len(STUDIES[study]["seeds"])
    return np.random.default_rng(seed).integers(0, n, size=(BOOT, n))


def estimate(values, indices):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) == indices.shape[1], "Unpaired bootstrap length")
    require(np.isfinite(values).all(), "Nonfinite/censored bootstrap values")
    draws = values[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(estimate=float(values.mean()), ci_low=float(lo), ci_high=float(hi),
                verdict="positive" if lo > 0 else "negative" if hi < 0 else "unresolved",
                n_positive=int(np.sum(values > 0)), n_negative=int(np.sum(values < 0)))


def condition(study, activation, ln, age, arm):
    return (study, activation, ln, age, arm)


def weighted_terms(*parts):
    out = defaultdict(float)
    for weight, terms in parts:
        for key, value in terms.items():
            out[key] += weight * value
    return {k: v for k, v in out.items() if v != 0}


def base_terms(study, activation, ln, age, estimand):
    return {condition(study, activation, ln, age, arm): coef
            for arm, coef in BASE[estimand].items()}


def contrast_plan():
    """Return explicit linear contrasts; signs never depend on significance."""
    plans = []

    def add(study, kind, name, terms, primary=False, **context):
        require(bool(terms) and abs(sum(terms.values())) < 1e-12, "Invalid contrast")
        require(all(k[0] == study for k in terms), "Cross-study contrast is not registered")
        plans.append(dict(study=study, model=STUDIES[study]["model"], kind=kind,
                          contrast=name, primary_U=primary, terms=terms, **context))

    for study, cfg in STUDIES.items():
        for act, ln, age in itertools.product(cfg["acts"], ("off", "on"), cfg["ages"]):
            for name in BASE:
                add(study, "within_regime", name, base_terms(study, act, ln, age, name),
                    activation=act, ln=ln, t_end=age, estimand=name)

    # The addendum registers activation/LN comparisons for both scale and zero-head benefits.
    for estimand in ("B_P", "Z_P"):
        for age in (1, 10):
            for ln, comparator in itertools.product(("off", "on"), ("R", "LR", "SILU")):
                terms = weighted_terms((1, base_terms("A", "ELU", ln, age, estimand)),
                                       (-1, base_terms("A", comparator, ln, age, estimand)))
                add("A", "activation", f"{estimand}:ELU-{comparator}", terms,
                    primary=age == 10 and estimand == "B_P", activation=f"ELU-{comparator}",
                    ln=ln, t_end=age, estimand=estimand)
            for act in ACTS_A:
                terms = weighted_terms((1, base_terms("A", act, "off", age, estimand)),
                                       (-1, base_terms("A", act, "on", age, estimand)))
                add("A", "LN", f"{estimand}:LNoff-LNon", terms,
                    primary=age == 10 and estimand == "B_P", activation=act,
                    ln="off-on", t_end=age, estimand=estimand)
            for comparator in ("R", "LR", "SILU"):
                terms = weighted_terms((1, base_terms("A", "ELU", "off", age, estimand)),
                                       (-1, base_terms("A", comparator, "off", age, estimand)),
                                       (-1, base_terms("A", "ELU", "on", age, estimand)),
                                       (1, base_terms("A", comparator, "on", age, estimand)))
                add("A", "activation_by_LN", f"{estimand}:(ELU-{comparator})x(off-on)", terms,
                    primary=age == 10 and estimand == "B_P", activation=f"ELU-{comparator}",
                    ln="off-on", t_end=age, estimand=estimand)
        for ln in ("off", "on"):
            for act in ACTS_A:
                terms = weighted_terms((1, base_terms("A", act, ln, 10, estimand)),
                                       (-1, base_terms("A", act, ln, 1, estimand)))
                add("A", "age", f"{estimand}:task10-task1", terms,
                    activation=act, ln=ln, t_end="10-1", estimand=estimand)
            for comparator in ("R", "LR", "SILU"):
                terms = weighted_terms((1, base_terms("A", "ELU", ln, 10, estimand)),
                                       (-1, base_terms("A", comparator, ln, 10, estimand)),
                                       (-1, base_terms("A", "ELU", ln, 1, estimand)),
                                       (1, base_terms("A", comparator, ln, 1, estimand)))
                add("A", "age_by_activation", f"{estimand}:(ELU-{comparator})x(task10-task1)",
                    terms, activation=f"ELU-{comparator}", ln=ln, t_end="10-1", estimand=estimand)
        for ln in ("off", "on"):
            terms = weighted_terms((1, base_terms("B", "ELU", ln, 30, estimand)),
                                   (-1, base_terms("B", "R", ln, 30, estimand)))
            add("B", "current_activation", f"{estimand}:ELU-R", terms,
                primary=estimand == "B_P", activation="ELU-R", ln=ln, t_end=30, estimand=estimand)
        for act in ("R", "ELU"):
            terms = weighted_terms((1, base_terms("B", act, "off", 30, estimand)),
                                   (-1, base_terms("B", act, "on", 30, estimand)))
            add("B", "LN_history", f"{estimand}:LNoff-LNon", terms,
                activation=act, ln="off-on", t_end=30, estimand=estimand)
        terms = weighted_terms((1, base_terms("B", "ELU", "off", 30, estimand)),
                               (-1, base_terms("B", "R", "off", 30, estimand)),
                               (-1, base_terms("B", "ELU", "on", 30, estimand)),
                               (1, base_terms("B", "R", "on", 30, estimand)))
        add("B", "current_activation_by_LN_history", f"{estimand}:(ELU-R)x(off-on)", terms,
            primary=estimand == "B_P", activation="ELU-R", ln="off-on", t_end=30, estimand=estimand)
    keys = [(p["study"], p["kind"], p["contrast"], p.get("activation"),
             p.get("ln"), p.get("t_end")) for p in plans]
    require(len(keys) == len(set(keys)), "Duplicate planned contrast")
    return plans


def paired_values(df, terms, metric, study):
    """Join all terms by the actual seed, never by input row order."""
    seeds = list(STUDIES[study]["seeds"])
    vectors, weights = [], []
    for (st, act, ln, age, arm), coef in terms.items():
        g = df[(df.study == st) & (df.activation == act) & (df.ln == ln) &
               (df.t_end == age) & (df.arm == arm)]
        require(len(g) == len(seeds) and not g.seed.duplicated().any() and
                set(g.seed) == set(seeds), f"Missing/duplicate paired seeds for {(st,act,ln,age,arm)}")
        vectors.append(g.set_index("seed").reindex(seeds)[metric].to_numpy(dtype=float))
        weights.append(coef)
    values = np.column_stack(vectors)
    complete = np.isfinite(values).all(axis=1)
    if metric not in T90:
        require(complete.all(), f"Nonfinite uncensored {metric}")
    return values @ np.asarray(weights), complete


def term_json(terms):
    return json.dumps([dict(zip(("study", "activation", "ln", "t_end", "arm"), key),
                            coefficient=coef) for key, coef in sorted(terms.items())], sort_keys=True)


def summarize_contrasts(df):
    rows, seed_rows = [], []
    indices = {st: bootstrap_indices(st) for st in STUDIES}
    for p in contrast_plan():
        for metric in METRICS:
            values, complete = paired_values(df, p["terms"], metric, p["study"])
            row = {k: v for k, v in p.items() if k != "terms"}
            row.update(metric=metric, primary=bool(p["primary_U"] and metric == "U"),
                       coefficients=term_json(p["terms"]), n_seed=len(values),
                       n_complete_pairs=int(complete.sum()), estimable=bool(complete.all()),
                       estimate=None, ci_low=None, ci_high=None, verdict="censored",
                       n_positive=None, n_negative=None)
            if complete.all():
                row.update(estimate(values, indices[p["study"]]))
            is_pp = metric in ("U", "U_acute", "U_late", "accuracy_end")
            for field in ("estimate", "ci_low", "ci_high"):
                row[field + "_pp"] = 100 * row[field] if is_pp and row[field] is not None else None
            rows.append(row)
            for seed, value, ok in zip(STUDIES[p["study"]]["seeds"], values, complete):
                seed_rows.append({k: v for k, v in row.items() if k in
                    ("study", "model", "kind", "contrast", "activation", "ln", "t_end", "estimand", "metric", "primary")}
                    | dict(seed=seed, value=float(value) if ok else None, complete=bool(ok)))
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def summarize_arms(df):
    rows = []
    indices = {st: bootstrap_indices(st) for st in STUDIES}
    for key, g in df.groupby(list(FACTORS), sort=True):
        st = key[0]
        require(set(g.seed) == set(STUDIES[st]["seeds"]) and not g.seed.duplicated().any(),
                "Unpaired arm mean")
        g = g.set_index("seed").reindex(STUDIES[st]["seeds"])
        for metric in METRICS:
            values = g[metric].to_numpy(dtype=float)
            observed = np.isfinite(values)
            require(observed.all() or metric in T90, "Nonfinite ordinary arm metric")
            row = dict(zip(FACTORS, key)) | dict(metric=metric, n_seed=len(values),
                n_observed=int(observed.sum()), n_censored=int((~observed).sum()),
                estimable=bool(observed.all()), estimate=None, ci_low=None, ci_high=None)
            if observed.all():
                row.update(estimate(values, indices[st]))
            rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
    return "\n".join(lines)


def summary_text(df, contrasts, validation):
    chunks = ["# ELU specificity: complete registered analysis",
        f"Validated {validation['actual']} full curves: Study A 1120 (10 paired seeds), Study B 140 (5 paired seeds).",
        "U is trapezoidal period-average fixed-training accuracy deficit. Positive B_S/B_P means "
        "the physical scale reduction improves average training accuracy; positive Z_S/Z_P means "
        "zero-head/body-memory-keep improves it. These are distinct interventions. "
        "Positive body_utility=U(Z_clear)-U(F0) means the learned body performs worse than "
        "the fresh-body reference under zero head and fresh Adam; it does not measure irreducible capacity.",
        "Each bootstrap draw resamples a whole seed cluster; the same 10000 draws are reused across "
        "arms, ages, LN levels and current activations within a study. Percentile 95% intervals are "
        "descriptive and unadjusted. Signed support requires the interval to lie entirely above/below "
        "zero; zero crossing is unresolved, not equivalence. The 1 pp reference is not an equivalence test.",
        "Study A identifies total activation/final-affine-LN training-regime effects including learned "
        "state formation. Study B identifies a persistent current-activation switch on ReLU-trained "
        "ViT histories, including compatibility with retained optimizer memory. Its all-five-LN "
        "history treatment differs from MLP final LN. No architecture-only effect, additive causal "
        "share, or exclusion of ELU saturation follows from these comparisons.",
        "U_acute covers [0,200], U_late covers [200,end]; Head edits and Adam resets are applied once at the fork; all parameters then learn jointly. Only the chosen activation/LN configuration persists, with no step200 release. "
        "U uses the entire 4000/1500-update horizon, and all endpoints share that horizon. "
        "CE area is the unnormalized integral; ce_auc_mean divides it by the horizon. "
        "T90 is the first saved observation ≥90%; sustained T90 requires all later saved observations "
        "to remain ≥90% and may have zero follow-up duration. Missing T90 is censored, never replaced "
        "by the budget; a censored arm mean/paired contrast is not estimated from a selected subset.",
        "## Primary U interactions (percentage points)"]
    for study in STUDIES:
        g = contrasts[(contrasts.study == study) & contrasts.primary].copy()
        g["estimate_pp_95ci"] = g.apply(lambda r:
            f"{r.estimate_pp:+.3f} [{r.ci_low_pp:+.3f}, {r.ci_high_pp:+.3f}]", axis=1)
        chunks.extend([f"### Study {study}", md_table(g,
            ["kind", "contrast", "activation", "ln", "t_end", "estimate_pp_95ci", "verdict"])])
    chunks.append("## Absolute zero-head/fresh-body performance")
    rows = []
    for keys, g in df[df.arm.isin(("Z_keep", "Z_clear", "F0"))].groupby(list(FACTORS), sort=True):
        rows.append(dict(zip(FACTORS, keys)) | dict(
            U_pp=f"{g.U.mean()*100:.3f}", endpoint_accuracy_pct=f"{g.accuracy_end.mean()*100:.3f}",
            T90_first_reached=f"{int((~g.T90_first_censored).sum())}/{len(g)}",
            T90_sustained_reached=f"{int((~g.T90_sustained_censored).sum())}/{len(g)}"))
    chunks.append(md_table(pd.DataFrame(rows), ["study", "activation", "ln", "t_end", "arm",
        "U_pp", "endpoint_accuracy_pct", "T90_first_reached", "T90_sustained_reached"]))
    chunks.append("All registered estimates and explicit linear coefficients are in contrasts.csv; "
        "seed_contrasts.csv exposes each paired difference. Arm means, censor counts and intervals "
        "are in arm_means.csv. seed_metrics.csv retains curve hashes and numeric diagnostics. "
        "validation.json records completeness and provenance consistency; it does not replace "
        "the independent source/archive byte-hash audit.")
    return "\n\n".join(chunks) + "\n"


def make_figures(df, out):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/elu_specificity_1112_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), constrained_layout=True)
    colors = {"off": "#2166ac", "on": "#b35806"}
    for col, study in enumerate(STUDIES):
        age = 10 if study == "A" else 30
        acts = ACTS_A if study == "A" else ("R", "ELU")
        for row, estimand in enumerate(("B_P", "Z_P")):
            ax = axes[row, col]
            for x, act in enumerate(acts):
                for ln, offset in (("off", -.13), ("on", .13)):
                    values, complete = paired_values(df, base_terms(study, act, ln, age, estimand), "U", study)
                    require(complete.all(), "Unexpected missing plot values")
                    e = estimate(values, bootstrap_indices(study))
                    xx = x + offset
                    ax.scatter(np.full(len(values), xx), values*100, s=13, alpha=.45, color=colors[ln])
                    ax.errorbar(xx, e["estimate"]*100,
                        yerr=[[100*(e["estimate"]-e["ci_low"])], [100*(e["ci_high"]-e["estimate"])]],
                        fmt="o", capsize=3, color=colors[ln], label=f"LN {ln}" if x == 0 else None)
            ax.axhline(0, color=".55", lw=.8)
            ax.set(xticks=range(len(acts)), xticklabels=acts, ylabel=f"{estimand} benefit (pp)",
                   title=f"A: MLP trained regime, task10" if study == "A" else
                         "B: ViT current switch, ReLU-trained task30")
            ax.legend(fontsize=9)
    fig.suptitle("P-conditioned benefits: physical scale vs zero head\nSeed points, means and paired-seed bootstrap 95% intervals")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"p_conditioned_benefits.{ext}", dpi=180)
    plt.close(fig)


def atomic_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", delete=False) as f:
        temp = Path(f.name)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run_analysis(raw, out, figures=True):
    records = load_records(raw)
    validation = validate_design(records)
    df = pd.DataFrame(metric_row(r) for r in records)
    contrasts, seed_contrasts = summarize_contrasts(df)
    arms = summarize_arms(df)
    validation.update(bootstrap_replicates=BOOT, bootstrap_cluster="seed",
        seed_clusters={st: list(cfg["seeds"]) for st, cfg in STUDIES.items()},
        n_contrast_estimates=len(contrasts), n_primary_U=int(contrasts.primary.sum()),
        n_censored_T90_contrasts=int((~contrasts.estimable).sum()),
        raw_manifest=[dict(path=r["_path"], sha256=r["_sha256"], key=list(record_key(r))) for r in records],
        analysis_sha256=sha256(__file__))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # No output is published before the full design and all calculations validate.
    for name, frame in (("seed_metrics.csv", df), ("contrasts.csv", contrasts),
                        ("seed_contrasts.csv", seed_contrasts), ("arm_means.csv", arms)):
        atomic_text(out/name, frame.to_csv(index=False))
    atomic_text(out/"summary.md", summary_text(df, contrasts, validation))
    if figures:
        make_figures(df, out)
    atomic_text(out/"validation.json", json.dumps(validation, indent=2, allow_nan=False)+"\n")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT/"results/elu_specificity_1112/raw")
    parser.add_argument("--out", type=Path, default=ROOT/"results/elu_specificity_1112")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    validation = run_analysis(args.raw, args.out, figures=not args.no_figures)
    print(json.dumps({k: v for k, v in validation.items() if k != "raw_manifest"}, indent=2))


if __name__ == "__main__":
    main()

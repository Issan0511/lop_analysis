"""Strict aggregation: incomplete or smoke shards never produce main verdicts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.resp_ee_0917.verdict import paired, spearman
from src import lc_complement_0917 as C


def load(folder, part, seed, arm=None):
    p = json.loads((folder / "provenance.json").read_text())
    if p["experiment"] != C.EXPERIMENT or p["part"] != part or p["status"] != "COMPLETE" or p["smoke"]:
        raise ValueError(f"Not a complete main shard: {folder}")
    if p["config"]["seed"] != seed or (arm is not None and p["config"]["arm"] != arm):
        raise ValueError(f"Wrong seed/arm: {folder}")
    if p["spec_sha256"] != C.sha(C.ROOT / C.SPEC) or p["git_dirty_code"]:
        raise ValueError(f"Unregistered or different specification: {folder}")
    if p["code_sha256"] != C.code_hashes(part):
        raise ValueError(f"Different implementation: aggregate from the run's commit: {folder}")
    df = pd.read_csv(folder / "rows.csv", float_precision="round_trip")
    if set(df.seed) != {seed} or (arm is not None and set(df.arm) != {arm}):
        raise ValueError("CSV identity differs from provenance")
    return df


def require_grid(df, cols, expected):
    observed = list(df[cols].itertuples(index=False, name=None))
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError(f"Duplicate, missing or unexpected rows in {cols}")


def c1(df):
    require_grid(df, ["seed", "arm", "layer"],
                 [(s, a, l) for s in range(10) for a in C.R.ARMS for l in (1, 2)])
    if not np.isfinite(df[["E", "lc_frac", "tr_frac", "zero_frac", "neff_abs"]]).all().all():
        raise ValueError("Nonfinite C1 primary metrics")
    rows = []
    for layer in (1, 2):
        d = df[df.layer == layer]
        means = d.groupby("arm").mean(numeric_only=True)
        for metric in ("lc_frac", "tr_frac", "zero_frac", "neff_abs"):
            r = spearman(means.E, means[metric])
            rows.append(dict(layer=layer, metric=metric, aggregation="26_arm_means",
                             rho=r if np.isfinite(r) else None))
            for seed, sd in d.groupby("seed"):
                r = spearman(sd.E, sd[metric])
                rows.append(dict(layer=layer, metric=metric, aggregation="within_seed", seed=int(seed),
                                 rho=r if np.isfinite(r) else None))
    # LC output masks must match exactly for these compensated interventions.
    for seed in range(10):
        for layer in (1, 2):
            d = df[(df.seed == seed) & (df.layer == layer)].set_index("arm")
            if d.loc["R2_10", "lc_n"] != d.loc["N10", "lc_n"]:
                raise ValueError("Matched logits did not preserve LC masks")
    return rows


def c1_gpu(df):
    from src import neff_pred_0917 as N
    require_grid(df, ["engine", "seed", "env", "act1", "act2", "task", "layer", "phase"],
                 [(engine, m["seed"], m["env"], *N.model_key(m), t, l, phase)
                  for engine, models in (("B", N.B.MODELS), ("L", N.L.MODELS)) for m in models
                  for t in range(1,151) for l in (1,2) for phase in ("start", "end")])
    if not np.isfinite(df[["lc_frac", "tr_frac", "zero_frac", "neff_abs"]]).all().all():
        raise ValueError("Nonfinite GPU metrics")
    selected = df[(df.engine == "B") & (df.env == "RL") & (df.act1 == "GELU") &
                  (df.task == 50) & (df.layer == 2) & (df.phase == "end")]
    rows = [dict(query="RL_GELU_l2_t50", seed=int(r.seed), A=float(r.A) if pd.notna(r.A) else None,
                 prediction_A_ge_point3=bool(r.A >= .3) if pd.notna(r.A) else None)
            for r in selected.itertuples()]
    for key, group in df[df.phase == "start"].groupby(["engine", "seed", "env", "act1", "act2", "layer"]):
        if not np.isfinite(group.E).all():
            raise ValueError("Missing future E for start field")
        for metric in ("lc_frac", "tr_frac", "zero_frac", "neff_abs"):
            rho = spearman(group.E, group[metric])
            rows.append(dict(zip(("engine", "seed", "env", "act1", "act2", "layer"), key),
                             query="within_trajectory_descriptive", metric=metric,
                             rho=rho if np.isfinite(rho) else None))
    return rows


def c2(df):
    require_grid(df, ["seed", "branch", "arm", "k"],
                 [(s, b, a, k) for s in range(10) for b in (2, 5) for a in C.SHOCK_ARMS for k in (1, 2)])
    if not np.isfinite(df.online_acc).all() or not df.finite.eq(True).all():
        raise ValueError("Nonfinite C2 run: no confirmatory verdict")
    rows = []
    for branch in (2, 5):
        for k in (1, 2):
            d = df[(df.branch == branch) & (df.k == k)].pivot(index="seed", columns="arm", values="online_acc")
            cis = {name: paired((d[a] - d[b]).to_numpy(), .95) for name, a, b in
                   (("rev_minus_N", "K_rev", "N"), ("hold_minus_rev", "K_hold2", "K_rev"),
                    ("sw_minus_mid", "K_sw2", "K_mid2"))}
            labels = []
            if cis["rev_minus_N"]["sign"] == "0" and cis["hold_minus_rev"]["sign"] == "-":
                labels.append("PERSISTENCE_MATTERS")
            if cis["sw_minus_mid"]["sign"] == "-":
                labels.append("TIMING_MATTERS")
            if cis["rev_minus_N"]["sign"] == "-":
                labels.append("SHOCK_HARMS")
            for name, ci in cis.items():
                rows.append(dict(branch=branch, k=k, tier="primary" if (branch,k)==(5,1) else "report",
                                 comparison=name, labels="|".join(labels) or "INCONCLUSIVE", **ci))
    return rows


def c3(df):
    require_grid(df, ["seed", "arm", "task"],
                 [(s, a, t) for s in range(3) for a in C.C3_ARMS for t in range(1, 51)])
    if not np.isfinite(df[["fit", "online_acc"]]).all().all():
        raise ValueError("Nonfinite C3 data")
    rows = []
    for (seed, arm), d in df.groupby(["seed", "arm"]):
        below = d[d.fit < .5].task
        rows.append(dict(seed=int(seed), arm=arm, late_fit=float(d[d.task >= 41].fit.mean()),
                         online_life=float(d.online_acc.mean()),
                         T_half=int(below.min()) if len(below) else None,
                         T_half_censored=not len(below)))
    late = pd.DataFrame(rows).pivot(index="seed", columns="arm", values="late_fit")
    slope = (late.S36 + late.E36 - late.E1 - late.C36) / 2
    cap = (late.C36 + late.E36 - late.E1 - late.S36) / 2
    for name, values in (("slope_effect", slope), ("cap_effect", cap), ("slope_minus_cap", slope-cap)):
        rows.append(dict(arm=name, tier="report_3_seeds", **paired(values.to_numpy(), .95)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part", choices=("c1", "c1-gpu", "c2", "c3"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    if args.part == "c1-gpu":
        p = json.loads((args.input / "provenance.json").read_text())
        if (p["experiment"] != C.EXPERIMENT or p["part"] != "c1-gpu" or p["smoke"] or
                p["status"] != "COMPLETE" or p["spec_sha256"] != C.sha(C.ROOT / C.SPEC) or p["git_dirty_code"]):
            raise ValueError("Not a complete registered GPU replay")
        if p["code_sha256"] != C.code_hashes("c1-gpu"):
            raise ValueError("GPU implementation differs; aggregate at the run's commit")
        df = pd.read_csv(args.input / "rows.csv", float_precision="round_trip")
    elif args.part == "c3":
        df = pd.concat([load(args.input / a / f"s{s}", "c3", s, a) for a in C.C3_ARMS for s in range(3)])
    else:
        df = pd.concat([load(args.input / f"s{s}", "c1-ee" if args.part == "c1" else "c2", s)
                        for s in range(10)])
    result = {"c1": c1, "c1-gpu": c1_gpu, "c2": c2, "c3": c3}[args.part](df)
    args.out.mkdir(parents=True, exist_ok=True)
    C.R.write_csv(args.out / "verdict.csv", result)
    C.dump(args.out / "verdict.json", result)
    table = pd.DataFrame(result).to_csv(index=False)
    caveat = ("C1: descriptive associations; no independence of the 26 arms is assumed."
              if args.part.startswith("c1") else "C2: an interval containing zero is not evidence of equivalence."
              if args.part == "c2" else "C3: native ELU/CELU in the LC benchmark protocol; 3 seeds; exploratory factorial effects.")
    (args.out / "summary.md").write_text(f"# {args.part.upper()}\n\n{caveat}\n\n```csv\n{table}```\n")


if __name__ == "__main__":
    main()

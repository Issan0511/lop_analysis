"""Preserve registered verdict and append the diagnosed numerical interpretation."""
import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import verify as V

OUT = V.ROOT/"results/aq_identity_0917"
diag = json.loads((OUT/"diagnosis.json").read_text())
checks = list(csv.DictReader((OUT/"checks.csv").open()))
matched = list(csv.DictReader((OUT/"diagnostic_checks.csv").open()))
primary = list(csv.DictReader((OUT/"verdict.csv").open()))
raw = Path(json.loads((OUT/"provenance.json").read_text())["raw_directory"])

old = subprocess.check_output(["git","show","0ba13e8:src/nets.py"],cwd=V.ROOT,text=True)
new = (V.ROOT/"src/nets.py").read_text()
def methods(s):
    c = next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=="VecMLPL")
    out = {}
    for m in c.body:
        if isinstance(m,ast.FunctionDef):
            out[m.name] = ast.dump(m,include_attributes=False)
            if m.name in ["act_fn","act_grad"]:
                for n in m.body:
                    if isinstance(n,ast.If) and any(q in ast.unparse(n.test) for q in
                        ["leaky_off_0","LEAKY_OFFSET","leaky_relu"]):
                        out[m.name+"/"+ast.unparse(n.test)] = ast.dump(n,include_attributes=False)
        if isinstance(m,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="LEAKY_OFFSET" for t in m.targets):
            out["LEAKY_OFFSET"] = ast.dump(m.value,include_attributes=False)
    return out
a,b = methods(old),methods(new)
needed = ["forward_layers","forward","grads_layers","grads","sgd_step_layers","sgd_step","LEAKY_OFFSET"]
needed += [k for k in a if "/" in k]
equiv = {k:a[k]==b.get(k) for k in needed}
assert all(equiv.values()), equiv
(OUT/"source_equivalence.json").write_text(json.dumps(dict(
    original_training_commit="0ba13e8",production_methods_and_selected_activation_branches=equiv),indent=2))

numerical = []
for dtype in ["float64","float32"]:
    err,denom,resolved = [],[],[]
    for path in sorted(raw.glob(f"*_{dtype}_frozen.npz")):
        with np.load(path) as z:
            for lr in V.LRS:
                p=f"lr{lr}_row_centered_"
                e=np.abs(z[p+"observed"]-z[p+"predicted"]).ravel()
                scale=(np.abs(z[p+"linear"])+z[p+"quadratic"]).ravel()
                mask=(np.abs(z[p+"predicted"])>z[p+"tol"]).ravel()
                err.append(e);denom.append(scale);resolved.append(mask)
    e,d,m=map(np.concatenate,[err,denom,resolved])
    ratio=np.divide(e,d,out=np.zeros_like(e),where=d>0)
    numerical.append(dict(dtype=dtype,max_term_scaled_error_all=float(ratio.max()),
        max_term_scaled_error_resolved=float(ratio[m].max()),
        p99_term_scaled_error_resolved=float(np.quantile(ratio[m],.99)),
        unresolved_count=int((~m).sum()),total_count=len(m)))
V.writecsv(OUT/"numerical_resolution.csv",numerical)

verdict = dict(registered_verdict=primary[0]["verdict"],diagnosis=diag["diagnosis"],
    original_failed_comparisons=diag["original_failures_reproduced"],
    matched_order_failed_comparisons=diag["matched_order_failures"],
    float64_identity="PASS",native_float32_budget="PASS",source_equivalence="PASS")
assert diag["diagnosis"]=="FORWARD_ROUNDING_ORDER"
assert all(int(r["failed"])==0 for r in checks if r["dtype"]=="float64")
assert all(int(r["failed"])==0 for r in checks if "/norm_budget" in r["check"])
V.writecsv(OUT/"diagnostic_verdict.csv",[verdict])

summary = (OUT/"summary.md").read_text().split("\n## Diagnostic follow-up")[0]
summary += "\n## Diagnostic follow-up (post-result, completed 2026-09-18)\n\n"
summary += ("The original aggregate **FAIL is retained**. All 100 failed scalar comparisons "
    "were reproduced. They came from float32 forward evaluation with elementwise sum "
    "versus the production einsum reduction, not from an incorrect A/Q identity. "
    "When autograd uses the production forward order, all comparisons pass at the "
    "unchanged tolerance; the production W gradient and autograd W gradient are identical. "
    "The two original forward paths differed in output by up to "
    f"{diag['forward_differences']['original_max_output_error']:.9g}; "
    "they had zero activation-gate disagreements. These are counts of scalar comparisons, "
    "not independent samples. The diagnosis was specified in a separate post-result addendum.\n\n")
summary += "### Individual float64 identities\n\n| check | scalar comparisons | maximum absolute error | failures |\n|---|---:|---:|---:|\n"
for r in checks:
    if r["dtype"]=="float64" and r["check"] in ["row_centered/A_vs_autograd","row_centered/Q_vs_autograd",
        "row_centered/A_SHBC","row_centered/norm_budget","row_centered/cumulative_budget","batch32/Q_vs_autograd"]:
        summary+=f"| {r['check']} | {r['count']} | {float(r['max_abs_error']):.9g} | {r['failed']} |\n"
summary += "\n### Float32 numerical resolution (fixed-state probes)\n\n"
summary += "| dtype | unresolved changes | total | max relative error over resolved updates | p99 relative error over resolved updates |\n|---|---:|---:|---:|---:|\n"
for r in numerical:
    summary+=f"| {r['dtype']} | {r['unresolved_count']} | {r['total_count']} | {r['max_term_scaled_error_resolved']:.6g} | {r['p99_term_scaled_error_resolved']:.6g} |\n"
summary += ("\nRelative error here divides by abs(linear contribution)+quadratic contribution, "
    "not the potentially cancelling net change. Float32 can suppress a mathematically "
    "nonzero tiny update entirely; those cases must not be interpreted as verified signs.\n\n"
    "The production forward, gradient and update methods and the selected leaky activation "
    "branches are AST-identical to the original training commit 0ba13e8. See source_equivalence.json.\n\n"
    "Diagnostic reproduction: python3 analysis/aq_identity_0917/diagnose.py\n\n"
    "Report/figure reproduction: python3 analysis/aq_identity_0917/report.py; "
    "python3 analysis/aq_identity_0917/plot.py\n")
(OUT/"summary.md").write_text(summary)
manifest=json.loads((OUT/"backup_manifest.json").read_text())
manifest=[m for m in manifest if not m["backup"].endswith(".log")]
for name in ["run.log","diagnosis.log"]:
    path=raw.parent/name
    manifest.append(dict(source="audit stdout/stderr",backup=str(path),
                         bytes=path.stat().st_size,sha256=V.sha(path)))
(OUT/"backup_manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(verdict))

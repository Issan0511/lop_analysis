"""Stage-1 lr selection (spec §4.2) -> lr_selection.csv.

Registered rule: among the lr values that do not diverge (finite for every
seed), take the one with the highest mean test accuracy over tasks 1-20;
ties go to the larger lr.
"""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
df = pd.read_csv(root / "per_task.csv")
prov = json.loads((root / "provenance.json").read_text())
seeds = set(prov["seeds"]); ntask = prov["n_tasks"]

rows = []
for (arm, lr), g in df.groupby(["arm", "lr"]):
    ok = g[g.acc.notna()]
    complete = {s for s, gg in ok.groupby("seed") if gg.task.max() == ntask}
    rows.append({"arm": arm, "lr": lr, "diverged": len(complete) < len(seeds),
                 "n_complete": len(complete),
                 "mean_acc_1_20": ok[ok.task <= 20].acc.mean() if complete else float("nan")})
sel = pd.DataFrame(rows)

out = []
for arm, g in sel.groupby("arm"):
    ok = g[~g.diverged]
    if ok.empty:
        out.append({"arm": arm, "chosen_lr": None, "why": "every lr diverged"}); continue
    best = ok.mean_acc_1_20.max()
    chosen = ok[ok.mean_acc_1_20 >= best - 1e-12].lr.max()     # tie -> larger lr
    out.append({"arm": arm, "chosen_lr": chosen, "mean_acc_1_20": best,
                "survived": ",".join(f"{x:g}" for x in sorted(ok.lr)),
                "diverged": ",".join(f"{x:g}" for x in sorted(g[g.diverged].lr))})
res = pd.DataFrame(out)
res.to_csv(root / "lr_selection.csv", index=False)

print(f"{'arm':>5} {'lr':>7} {'acc(1-20)':>10} {'n_ok':>5}  status")
print("-" * 56)
for _, r in sel.sort_values(["arm", "lr"]).iterrows():
    st = f"DIVERGED ({r.n_complete}/{len(seeds)} seeds finished)" if r.diverged else "ok"
    a = "nan" if r.mean_acc_1_20 != r.mean_acc_1_20 else f"{r.mean_acc_1_20:.4f}"
    print(f"{r.arm:>5} {r.lr:>7g} {a:>10} {int(r.n_complete):>5}  {st}")
print("\n=== §4.2 selection ===")
for _, r in res.iterrows():
    print(f"  {r.arm:>4} -> lr = {r.chosen_lr}   (survived: {r.get('survived','-')}"
          f" | diverged: {r.get('diverged','') or 'none'})")
print(f"\nwrote {root}/lr_selection.csv")

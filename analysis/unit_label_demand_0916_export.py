"""Export preregistered per-unit group summaries without recomputing diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import torch
from src.unit_label_demand_0916_report import metrics, FI, GROUPS
from src import relu_gelu_silu_rl_0914 as B

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/unit_label_demand_0916"
ap=argparse.ArgumentParser()
ap.add_argument("--raw",type=Path,default=OUT/"raw")
raw=ap.parse_args().raw
with np.load(raw/"statistics.npz") as f:
    pooled=f["values"][f["steps"]==0].sum(0)
models=torch.load(raw/"inputs.pt",weights_only=True)["models"]
rows=[]
for j,m in enumerate(models):
    for l in range(2):
        for i in range(100):
            n_all=pooled[j,l,i,:,FI["n"]].sum()
            for g,group in enumerate(GROUPS):
                rows.append(dict(**m,window="t2-10",step=0,layer=l+1,unit=i,group=group,
                                 **metrics(pooled[j,l,i,g],n_all)))
B.csvwrite(OUT/"unit_group_summary.csv",rows)
print(f"Exported {len(rows)} unit/group rows")

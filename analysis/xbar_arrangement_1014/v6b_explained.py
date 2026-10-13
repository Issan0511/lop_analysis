"""V6b (verifier): how much of the actual N_i(T) variation do the surrogate mean series explain?  Pooled over alive phase-2 (i, T) of the
targets (>= 30 alive tasks): var ratio Var(Nbar)/Var(N) vs R^2 = Pearson(N, Nbar)^2 and the residual-variance share 1 - Var(N - Nbar)/Var(N);
per target (median): the same within target over T.  Agent B arrays, iso and isoc."""
import json, os
import numpy as np
from v_common import RES, OBS

def main():
    out = {}
    for cond in ("a01s0", "a03s0", "a05s0", "ELU_s0_E1", "ELU_s0_E2", "ELU_s1_E1", "ELU_s1_E2"):
        out[cond] = {}
        for mode in ("iso", "isoc"):
            b = np.load(os.path.join(OBS, "xbar_arrangement_1014/agent_b", f"b_arrays_{mode}_{cond}.npz")); N = b["N"]; M = b[f"N{mode}_mean"]
            xs, ys, vr, r2, res = [], [], [], [], []
            for i in range(N.shape[1]):
                sel = N[:, i] >= 0
                if sel.sum() < 30: continue
                n = N[sel, i].astype(float); mm = M[sel, i].astype(float); xs.append(n); ys.append(mm)
                if n.var() > 0 and mm.var() > 0:
                    vr.append(mm.var(ddof=1) / n.var(ddof=1)); r2.append(np.corrcoef(n, mm)[0, 1] ** 2); res.append(1 - (n - mm).var() / n.var())
            x = np.concatenate(xs); y = np.concatenate(ys)
            out[cond][mode] = dict(pooled_var_ratio=float(y.var() / x.var()), pooled_R2=float(np.corrcoef(x, y)[0, 1] ** 2), pooled_1_minus_resvar=float(1 - (x - y).var() / x.var()),
                                   target_median_var_ratio=float(np.median(vr)), target_median_R2=float(np.median(r2)), target_median_1_minus_resvar=float(np.median(res)))
            print(cond, mode, {k: round(v, 3) for k, v in out[cond][mode].items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "v6b_explained.json"), "w"), indent=1)

if __name__ == "__main__":
    main()

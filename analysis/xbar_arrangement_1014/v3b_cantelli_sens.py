"""V3b (verifier): sensitivity of the parent's per-state Cantelli rho_model to a multiplicative bias of the model mean c(T)
(c -> s c, s = 1, 1.5, 2, 3), 8 leaky conditions with n_beta; seed 0 uses full + grid states (all phase-2 states)."""
import json, os, re
import numpy as np
from v3_cantelli import nbeta_files, THR
from v_common import RES, OBS
THR = dict(THR, a02=0.14)

def main():
    out = {}
    for cond in ("a01s0", "a03s0", "a05s0", "a01s1", "a02s0", "a02s1", "a03s1", "a05s1"):
        files = nbeta_files(cond, True); nb = {T: {int(r["i"]): float(r["n_b"]) for r in json.load(open(f))["real"]} for T, f in files.items()}
        b = np.load(os.path.join(OBS, "xbar_arrangement_1014/agent_b", f"b_arrays_iso_{cond}.npz")); N = b["N"]; C = b["Niso_mean"]; TL = [int(t) for t in b["T_list"]]
        thr = THR[re.match(r"(a\d+)", cond).group(1)]; res = {}
        for s in (1.0, 1.5, 2.0, 3.0):
            rho = []
            for i in range(N.shape[1]):
                Ts = [T for T in TL if T in nb and i in nb[T] and N[TL.index(T), i] >= 0]
                if len(Ts) < 30: continue
                c = s * C[[TL.index(T) for T in Ts], i]; n = np.array([nb[T][i] for T in Ts])
                rho.append(np.where(c < n, c / (c + (n - c) ** 2), 1.0).mean())
            rho = np.array(rho); res[str(s)] = dict(median=float(np.median(rho)), max=float(rho.max()), n_above=int((rho > thr).sum()), n=len(rho))
        out[cond] = dict(thr=thr, n_states=len(files), **res)
        print(cond, len(files), {k: (round(v["median"], 4), round(v["max"], 4), v["n_above"]) for k, v in res.items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "v3b_cantelli_sens.json"), "w"), indent=1)

if __name__ == "__main__":
    main()

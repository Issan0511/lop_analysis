"""xbar_arrangement_1014 agent B: per-condition arrays for the coordinator (from b_surr_<tag>.npz written by b_strat_null.py).
b_arrays_<tag>.npz keys (rows = phase-2 states T = 11..100, columns = all 100 units; -1 where the unit is not alive at T):
  N, C (int16, actual); Nstrat_mean, Cstrat_mean (float64, mean over the 20 strat draws); Nstrat_draws (int16, [r, T, i]);
  Nflat_mean, Cflat_mean (float64, mean over the 20 flat draws); T_list (int, 11..100)."""
import os
import numpy as np
BIG = "/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b"
TAGS = ["a003s0", "a003s1", "a01s0", "a01s1", "a02s0", "a02s1", "a03s0", "a03s1", "a05s0", "a05s1", "ELU_s0_E1", "ELU_s0_E2", "ELU_s1_E1", "ELU_s1_E2"]
for t in TAGS:
    z = np.load(os.path.join(BIG, f"b_surr_{t}.npz"))
    N = z["N_act"].astype(np.int16); C = z["C_act"].astype(np.int16); dead = N < 0
    for k in ("Ns_strat", "Cs_strat", "Ns_flat", "Cs_flat"):          # surrogate alive pattern = target's actual alive pattern
        assert ((z[k] < 0) == dead[None]).all(), (t, k)
    def mean(k):
        m = z[k].astype(np.float64).mean(0); m[dead] = -1.0; return m
    out = dict(N=N, C=C, Nstrat_mean=mean("Ns_strat"), Cstrat_mean=mean("Cs_strat"), Nstrat_draws=z["Ns_strat"].astype(np.int16),
               Nflat_mean=mean("Ns_flat"), Cflat_mean=mean("Cs_flat"), T_list=np.arange(11, 101))
    p = os.path.join(BIG, f"b_arrays_{t}.npz"); np.savez_compressed(p, **out)
    print(t, {k: v.shape for k, v in out.items()}, os.path.getsize(p))

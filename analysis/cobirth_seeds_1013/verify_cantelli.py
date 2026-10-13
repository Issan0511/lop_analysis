"""TASK V (cobirth_seeds_1013, verifier, 1014): Cantelli inputs and the two sentences (B') and (N*') per condition, recomputed from the
verifier's own counts (verify/vcounts_<tag>.npz, bit-identical to the agents' N) and the p172c n_beta files (generalized copies, checked
bit-identical to the original p172c logic at one state per copy). Convention: M1, T = 1..100, state end0[T-1]; p172c file 't' = T + 1.
For each condition:
  states      : the T at which an n_beta file exists (state-matched comparisons use only those T)
  (N*')       : min over targets of (min_T n_beta(i,T) - mu_i)  ['per-target min' form, README section 4.1]  and
                min over computed (i,T) with T in 11..100 of (n_beta(i,T) - mu_i)  [state-matched]
  emp         : #(i,T) at computed states with N_i(T) >= n_beta(i,T); max N/n_beta
  cantelli    : state-matched per-(i,T) max of s_i^2/(s_i^2 + (n_beta - mu_i)^2) (mu_i, s_i^2 ddof=1 from the same 90-task N series)
  (B')+(N*')  : max over targets of Fbar mu/(Fbar mu + (n*_i - mu)^2) with n*_i = per-target min n_beta and Fbar = measured max_i F_i, and Fbar = 6.4
  P177d check : a0.5 seed 1 at T 25 (t 26): N_12, n_beta(12), A(12)."""
import os, json, glob, re
import numpy as np
V = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013"
NB0 = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/block_1003/stage14_1007_h4/agent_bound"
THR = dict(a01=0.14, a03=0.11, a05=0.08)

def load_nb(path):
    j = json.load(open(path)); return {int(r["i"]): (float(r["n_b"]), float(r["A"]), float(r["self"])) for r in j["real"]}

def files(tag):
    if tag in ("a01s0", "a03s0", "a05s0"):
        arm = tag[:3]; out = {}
        for f in glob.glob(f"{V}/nbeta_time/grid/p172c_{arm}_T*_real.json") + glob.glob(f"{V}/nbeta_time/full/p172c_{arm}_T*_real_slim.json"):
            out[int(re.search(r"_T(\d+)_", f).group(1))] = f
        return out
    if tag in ("a01s1", "a03s1", "a05s1"):
        return {int(re.search(r"_t(\d+)_", f).group(1)) - 1: f for f in glob.glob(f"{V}/seed1/p172c/p172c_{tag}_t*_real.json")}
    m = {"a003s0": "a0.03_s0", "a003s1": "a0.03_s1", "a02s0": "a0.2_s0", "a02s1": "a0.2_s1"}[tag]
    return {int(re.search(r"_T(\d+)_", f).group(1)): f for f in glob.glob(f"{V}/slopes/p172c/p172c_{m}_T*_real.json")}

res = {}
for tag in ["a01s0", "a03s0", "a05s0", "a01s1", "a03s1", "a05s1", "a003s0", "a003s1", "a02s0", "a02s1"]:
    st = json.load(open(f"{V}/verify/vstats_{tag}.json")); pt = st["per_target"]; ids = pt["i"]; mu = dict(zip(ids, pt["mu"])); var = dict(zip(ids, pt["var"]))
    F = {i: var[i] / mu[i] for i in ids if mu[i] > 0}; Fbar = max(F.values())
    N = np.load(f"{V}/verify/vcounts_{tag}.npz")["N"]
    fs = files(tag); NB = {T: load_nb(f) for T, f in sorted(fs.items())}
    Ts = sorted(NB); Ts2 = [T for T in Ts if 11 <= T <= 100]
    allnb = [(T, i, v[0]) for T in Ts for i, v in NB[T].items()]
    nbmin_i = {}
    for T, i, nb in allnb:
        if i in mu: nbmin_i[i] = min(nbmin_i.get(i, np.inf), nb)
    pertarget_min_gap = min((nbmin_i[i] - mu[i], i) for i in nbmin_i)
    sm = []; emp = 0; ratio = 0.0; cmax = (0, None)
    for T in Ts2:
        for i, (nb, A, s) in NB[T].items():
            if i not in mu: continue
            n = N[T - 1, i]
            if n < 0: continue
            sm.append((nb - mu[i], i, T)); emp += int(n >= nb); ratio = max(ratio, n / nb)
            c = var[i] / (var[i] + (nb - mu[i]) ** 2) if nb > mu[i] else 1.0
            if c > cmax[0]: cmax = (c, (i, T, nb, mu[i], var[i]))
    def bn(Fb):
        best = (0, None)
        for i in nbmin_i:
            v = Fb * mu[i]; c = v / (v + (nbmin_i[i] - mu[i]) ** 2) if nbmin_i[i] > mu[i] else 1.0
            if c > best[0]: best = (c, i)
        return best
    holds = [(T, i) for T in Ts for i, (nb, A, s) in NB[T].items() if A <= 0 < s]
    res[tag] = dict(states=Ts, n_states=len(Ts), Fbar_measured=Fbar, arm_min_nbeta=min(x[2] for x in allnb), arm_min_at=[(T, i) for T, i, nb in allnb if nb == min(x[2] for x in allnb)],
                    Nstar_pertarget_min_gap=pertarget_min_gap, Nstar_state_matched_min_gap=min(sm), emp_N_ge_nbeta_state_matched=emp, n_state_matched=len(sm), max_N_over_nbeta=ratio,
                    cantelli_state_matched_max=cmax, BN_Fbar_measured=bn(Fbar), BN_Fbar_6p4=bn(6.4), holds_A_le0_lt_self=holds)
    r = res[tag]
    print(f"{tag}: {len(Ts)} states | Fbar(meas) {Fbar:.3f} | arm min n_beta {r['arm_min_nbeta']:.3f} at {r['arm_min_at']} | (N*') per-target min gap {pertarget_min_gap[0]:.3f} (u{pertarget_min_gap[1]}) | state-matched min gap {r['Nstar_state_matched_min_gap'][0]:.3f} (u{r['Nstar_state_matched_min_gap'][1]} T{r['Nstar_state_matched_min_gap'][2]}) | emp N>=n_beta {emp}/{len(sm)} max N/n_beta {ratio:.3f} | state-matched Cantelli max {cmax[0]:.4f} {cmax[1]} | B'+N*' Fbar=meas {r['BN_Fbar_measured'][0]:.4f} (u{r['BN_Fbar_measured'][1]}) Fbar=6.4 {r['BN_Fbar_6p4'][0]:.4f} | holds {holds}")
# P177d hold, a0.5 seed 1, original t 26 = T 25
Nv = np.load(f"{V}/verify/vcounts_a05s1.npz"); Nh = Nv["N"]; kh = Nv["k"]
pj = load_nb(f"{V}/verify/p172c_orig/p172c_V_a05s1_t26_real.json")
st = json.load(open(f"{V}/verify/vstats_a05s1.json")); mu12 = dict(zip(st["per_target"]["i"], st["per_target"]["mu"]))[12]
res["P177d_a05s1_T25_u12"] = dict(N_T24=int(Nh[23, 12]), N_T25=int(Nh[24, 12]), N_T26=int(Nh[25, 12]), k_T25=int(kh[24, 12]), n_beta=pj[12][0], A=pj[12][1], self=pj[12][2], mu12=mu12)
print("P177d a0.5 s1 T25 (t26) u12:", res["P177d_a05s1_T25_u12"])
json.dump(res, open(f"{V}/verify/verify_cantelli.json", "w"), indent=1, default=str)

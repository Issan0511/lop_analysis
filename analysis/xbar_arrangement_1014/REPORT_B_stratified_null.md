# REPORT B: brightness-stratified permutation null for N_i and C_i (xbar_arrangement_1014, agent B)

2026-10-14. Box: 784-100-10, 1,200 fixed inputs, random labels, 100 tasks, saved task-end weights only (no training). All numbers [実測] unless marked. Scripts: `analysis/xbar_arrangement_1014/b_strat_null.py` (per condition), `b_report.py` (this file's tables), `b_jobs.txt` (the 14 conditions). JSON: `results/xbar_arrangement_1014/b_cond_<tag>.json` (per condition, incl. per-target arrays), `b_summary.json` (all tables). Count arrays (actual and 20+20 surrogates, phase 2): `/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b/b_surr_<tag>.npz`.

## 1. Definitions (copied from `proj_004_drift/analysis/cobirth_seeds_1013/verify_counts.py`, function `counts`, verbatim)

- ROOT = `obsidian-research-data/push_direction_proof_0930`; X = `load_X(d['subset'])` (1200×784, /255); Xt = [X, 1]; K_n = Xt_n·mean(Xt) (> 0, asserted).
- State T = 1..100 (M1) = `end0[T−1]`, `end1[T−1]`; z_in = w_i·x_n + b_i; O_i = {n : z_in > 0}; k_i = |O_i|; alive = k_i > 0.
- Threshold inputs T_i = top-3 of a score (stable argsort): leaky / E1: log K_n − |x| − 2 log(1 + e^{−|x|}) − log τ_i with x = z/τ_i, τ_i = 0.1 sd_n(z_in) + 1e−12; E2: log K_n + z_in on z_in < 0, −∞ on open inputs.
- Bridge j → i (loose signature): j ≠ i, both alive, |O_j ∩ T_i| ≥ 2 and |O_j ∩ O_i| ≥ ⌈k_i/2⌉. N_i = number of bridges. C_i = #{j ≠ i alive : |O_j ∩ T_i| ≥ 2} (threshold pool; N_i ≤ C_i).
- Phase 2 = T 11..100 (90 states). Targets = units with ≥ 30 alive phase-2 tasks (100 in all 14 conditions). μ_i = mean of N_i(T) over the target's alive phase-2 tasks; σ_i² ddof = 1; F_i = σ_i²/μ_i (undefined if μ_i = 0; dropped from medians, counts given). Pooled Fano = var(ddof=1)/mean over all targets' alive phase-2 tasks. max N and #(N ≥ 10) over the same pool. Same statistics for C_i. The cobirth_seeds `C_fano` is the population-variance pooled Fano over all alive phase-2 (i, T) (identical pool here).
- Conditions (14): leaky `LR_ref_s{0,1}` (a = 0.1, `review/opus_bwdfloor/out/`), `LRa{0.03,0.2,0.3,0.5}_s{0,1}` (`review/opus_lrslope/out/`), gate leaky; ELU `ELU_ref_s{0,1}` (`review/opus_bwdfloor/out/`) with E1 and with E2.

**Surrogates (new).** K-deciles = rank split of K over the 1,200 inputs into 10 blocks of 120. At each phase-2 state, for every unit the gate column o_j = 1[z_j· > 0] is replaced by a random permutation: **strat** = independently within each K-decile (preserves the unit's open count per decile); **flat** = over all 1,200 inputs (preserves k_j only). One permutation per (unit, state, surrogate), shared by all targets for which that unit is a candidate. The target's own T_i, O_i, k_i stay actual; the target is never its own candidate. R = 20 surrogates per null, generator `np.random.default_rng(1000 + r)` per (null, r), drawn state by state T = 11..100 (strat and flat use separate generators with the same seeds). Preservation asserted at r = 0, T = 11 (column sums; per-decile column sums for strat). Each surrogate gives a full phase-2 series N_i^(r)(T), C_i^(r)(T); the statistics above are computed per surrogate; tables give median [5%, 95%] over the 20 surrogates (np.quantile, linear).

Criterion used in the verdicts: the null 'reproduces' a pooled statistic when the actual value lies inside the [5%, 95%] range of its 20 surrogate values.

## 2. Reproduction check against cobirth_seeds_1013 (verifier `vcounts_<tag>.npz`, `vstats_<tag>.json`; summary in `proj_004_drift/results/cobirth_seeds_1013/verify.json`)

| condition | N cells equal (100×100) | C equal (alive cells) | k equal | F_i median own / ref | pooled Fano N own / ref | C_fano own / ref | max N own/ref | N≥10 own/ref | verdict |
|---|---|---|---|---|---|---|---|---|---|
| a003s0 | True (0 diff) | True | True | 1.3500 / 1.3500 | 1.5870 / 1.5870 | 1.5282 / 1.5282 | 6/6 | 0/0 | MATCH |
| a003s1 | True (0 diff) | True | True | 1.2735 / 1.2735 | 1.6442 / 1.6442 | 1.5220 / 1.5220 | 7/7 | 0/0 | MATCH |
| a01s0 | True (0 diff) | True | True | 1.3846 / 1.3846 | 1.6312 / 1.6312 | 1.4718 / 1.4718 | 7/7 | 0/0 | MATCH |
| a01s1 | True (0 diff) | True | True | 1.3679 / 1.3679 | 1.5585 / 1.5585 | 1.4869 / 1.4869 | 6/6 | 0/0 | MATCH |
| a02s0 | True (0 diff) | True | True | 1.6467 / 1.6467 | 2.0341 / 2.0341 | 1.7908 / 1.7908 | 11/11 | 2/2 | MATCH |
| a02s1 | True (0 diff) | True | True | 1.4369 / 1.4369 | 1.7106 / 1.7106 | 1.5212 / 1.5212 | 8/8 | 0/0 | MATCH |
| a03s0 | True (0 diff) | True | True | 1.6425 / 1.6425 | 1.9657 / 1.9657 | 1.9355 / 1.9355 | 14/14 | 3/3 | MATCH |
| a03s1 | True (0 diff) | True | True | 1.5170 / 1.5170 | 1.7543 / 1.7543 | 1.6373 / 1.6373 | 9/9 | 0/0 | MATCH |
| a05s0 | True (0 diff) | True | True | 1.3285 / 1.3285 | 1.3902 / 1.3902 | 1.7828 / 1.7828 | 10/10 | 1/1 | MATCH |
| a05s1 | True (0 diff) | True | True | 1.3564 / 1.3564 | 1.4397 / 1.4397 | 1.6435 / 1.6435 | 10/10 | 1/1 | MATCH |
| ELU_s0_E1 | True (0 diff) | True | True | 2.8513 / 2.8513 | 3.6201 / 3.6201 | 3.9033 / 3.9033 | 18/18 | 32/32 | MATCH |
| ELU_s0_E2 | True (0 diff) | True | True | 1.7738 / 1.7738 | 2.2176 / 2.2176 | 3.6226 / 3.6226 | 11/11 | 1/1 | MATCH |
| ELU_s1_E1 | True (0 diff) | True | True | 3.9862 / 3.9862 | 4.8656 / 4.8656 | 5.0706 / 5.0706 | 28/28 | 80/80 | MATCH |
| ELU_s1_E2 | True (0 diff) | True | True | 2.2423 / 2.2423 | 2.8632 / 2.8632 | 4.1939 / 4.1939 | 14/14 | 10/10 | MATCH |

Ranges reproduced: N_i Fano median leaky 1.27–1.65 (ref 1.27–1.65), ELU 1.77–3.99 (ref 1.77–3.99); C_i pooled Fano leaky 1.47–1.94 (ref 1.47–1.94), ELU 3.62–5.07 (ref 3.62–5.07). **14/14 MATCH.**

## 2b. Verdicts (criterion: actual inside the [5%, 95%] range of the 20 surrogate values; ranges are over conditions, surrogate values are medians over 20)

- **strat reproduces actual pooled Fano of C_i: no** (0/14 in band). actual leaky 1.47–1.94, ELU 3.62–5.07; strat leaky 1.02–1.14, ELU 1.17–1.20; flat leaky 0.98–1.00, ELU 1.07–1.09 (0/14 in band).
- **strat reproduces actual median F_i of C_i: no** (0/14 in band). actual leaky 1.33–1.80, ELU 3.21–4.50; strat leaky 0.98–1.12, ELU 1.10–1.12; flat leaky 0.95–1.00, ELU 1.01–1.04 (0/14 in band).
- **strat reproduces actual pooled Fano of N_i: no** (1/14 in band). actual leaky 1.39–2.03, ELU 2.22–4.87; strat leaky 1.00–1.73, ELU 1.06–1.31; flat leaky 0.98–1.49, ELU 1.00–1.06 (0/14 in band).
- **strat reproduces actual median F_i of N_i: no** (0/14 in band). actual leaky 1.27–1.65, ELU 1.77–3.99; strat leaky 0.94–1.03, ELU 1.00–1.00; flat leaky 0.92–1.00, ELU 1.00–1.00 (0/14 in band).
- **strat reproduces μ_i of N_i: no** (pooled mean in band 0/14; strat/actual pooled mean leaky 0.07–0.23, ELU 0.01–0.04; per-target μ_i inside the strat band 0.00–0.09 of targets). flat/actual leaky 0.02–0.23, ELU 0.00–0.01.
- **strat reproduces μ_i of C_i: no** (pooled mean in band 0/14; strat/actual pooled mean leaky 0.12–0.93, ELU 0.07–0.11; per-target μ_i inside the strat band 0.00–0.39 of targets). flat/actual leaky 0.07–1.39, ELU 0.05–0.06.
- **flat does not reproduce** pooled Fano (N 0/14, C 0/14), μ (N 0/14, C 0/14), max N 0/14, #N≥10 in band 6/14 (all 6 are conditions with actual 0 and surrogate 0). flat pooled Fano of C: 0.98–1.09 (cobirth null U: 0.98–1.09).
- **common-cause correlation** (Spearman over (i, T), actual vs strat-mean): N leaky 0.20–0.44, ELU 0.17–0.41; C leaky 0.11–0.43, ELU 0.17–0.33. Median Var_T(N̄^strat)/Var_T(N): leaky 0.0036–0.0659, ELU 0.0002–0.0038; for C: leaky 0.0073–0.1885, ELU 0.0033–0.0076. flat Spearman: N 0.11–0.42, C -0.06–0.25.
- **B4 (ELU)**: strat does not reproduce the larger ELU Fano. N pooled Fano actual 2.22–4.87 vs strat 1.06–1.31; C pooled Fano actual 3.62–5.07 vs strat 1.17–1.20 (leaky strat 1.02–1.14). ELU max N actual 18, 11, 28, 14 vs strat medians 3, 2, 3, 2; #N≥10 actual 32, 1, 80, 10 vs strat 0 in all 80 surrogates.

**Flags (numbers only).**

1. The only in-band pooled statistic with nonzero actual value: N pooled Fano at a03s1 (actual 1.754, strat 1.732 [1.589, 1.804]); at a03s0 strat 1.641 [1.549, 1.784] vs actual 1.966. Its strat pooled mean is 0.041 vs actual 0.266.
2. strat N means are 0.07–0.23× actual (leaky) and 0.01–0.04× (ELU); strat F_i medians ≈ 1.000 coincide with 0/1-only series (share of targets whose strat series is 0/1-only with μ_i > 0: 0.17–0.74; targets with strat μ_i = 0 are dropped from the F_i median) [恒等式, see note in §3].
3. flat C mean exceeds actual at a = 0.5 (9.50, 9.60 vs 6.89, 6.91) and #(C ≥ 10) flat 4287, 4398 vs actual 1818, 1802; T_i share in decile 9: a0.5 0.727, 0.702; a0.03 0.030, 0.023; occupancy d9/d0 from 0.042 (ELU/a0.03) to 0.554 (a0.5).
4. Closest strat C mean: a = 0.5 (strat/actual 0.932, 0.923; per-target μ_i inside the strat band 0.39, 0.35); its C pooled Fano is still 1.092, 1.096 vs actual 1.783, 1.644.
5. Excess index (F_pool − 1)/μ_pool of C, strat vs actual: ELU_s0_E1 1.226 [0.999, 1.402] vs 1.476; ELU_s0_E2 0.750 [0.649, 0.954] vs 1.259; ELU_s1_E1 1.239 [1.037, 1.462] vs 1.838; ELU_s1_E2 0.823 [0.672, 1.013] vs 1.451; for N the strat excess index has very wide ranges when μ ≈ 0 (e.g. ELU_s1_E2 20.847 [-0.959, 64.000]), so it is not used in the verdicts.
6. ELU E2 flat: F_i undefined for 35 and 30 targets (surrogate μ_i = 0 in all 20 flat draws); B2 fractions use the remaining targets.

## 3a. B1 — N_i (loose bridges): actual vs strat vs flat (median [5%, 95%] over 20 surrogates)

| condition | F_i median: actual | strat | flat | pooled Fano: actual | strat | flat |
|---|---|---|---|---|---|---|
| a003s0 | 1.350 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.587 | 1.030 [0.990, 1.091] | 0.996 [0.996, 1.092] |
| a003s1 | 1.274 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.644 | 1.040 [0.992, 1.100] | 0.997 [0.996, 1.103] |
| a01s0 | 1.385 | 0.988 [0.987, 1.000] | 1.000 [0.988, 1.000] | 1.631 | 1.166 [1.089, 1.218] | 1.067 [1.024, 1.127] |
| a01s1 | 1.368 | 0.988 [0.987, 1.000] | 1.000 [0.988, 1.000] | 1.559 | 1.115 [1.071, 1.176] | 1.075 [1.013, 1.208] |
| a02s0 | 1.647 | 1.000 [1.000, 1.115] | 1.000 [0.989, 1.000] | 2.034 | 1.382 [1.293, 1.481] | 1.250 [1.166, 1.369] |
| a02s1 | 1.437 | 1.033 [1.000, 1.227] | 1.000 [0.989, 1.000] | 1.711 | 1.427 [1.376, 1.544] | 1.266 [1.190, 1.326] |
| a03s0 | 1.643 | 1.000 [1.000, 1.014] | 1.000 [1.000, 1.000] | 1.966 | 1.641 [1.549, 1.784] | 1.468 [1.338, 1.596] |
| a03s1 | 1.517 | 1.000 [1.000, 1.282] | 1.000 [1.000, 1.000] | 1.754 | 1.732 [1.589, 1.804] | 1.486 [1.365, 1.681] |
| a05s0 | 1.329 | 0.939 [0.907, 0.978] | 0.921 [0.899, 0.958] | 1.390 | 1.004 [0.976, 1.035] | 0.984 [0.960, 1.015] |
| a05s1 | 1.356 | 0.989 [0.973, 1.022] | 0.995 [0.977, 1.016] | 1.440 | 1.023 [0.996, 1.051] | 1.022 [1.000, 1.045] |
| ELU_s0_E1 | 2.851 | 1.000 [0.994, 1.000] | 1.000 [1.000, 1.000] | 3.620 | 1.308 [1.178, 1.397] | 1.056 [0.993, 1.133] |
| ELU_s0_E2 | 1.774 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 2.218 | 1.058 [0.997, 1.120] | 0.999 [0.999, 1.010] |
| ELU_s1_E1 | 3.986 | 1.000 [0.989, 1.000] | 1.000 [1.000, 1.000] | 4.866 | 1.291 [1.163, 1.384] | 1.064 [0.994, 1.157] |
| ELU_s1_E2 | 2.242 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 2.863 | 1.068 [0.997, 1.240] | 0.999 [0.999, 1.011] |

| condition | μ_i median: actual | strat | flat | pooled mean: actual | strat | flat |
|---|---|---|---|---|---|---|
| a003s0 | 0.124 | 0.000 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.151 | 0.011 [0.009, 0.013] | 0.004 [0.003, 0.005] |
| a003s1 | 0.110 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.127 | 0.008 [0.006, 0.010] | 0.003 [0.002, 0.004] |
| a01s0 | 0.179 | 0.026 [0.025, 0.035] | 0.012 [0.012, 0.013] | 0.186 | 0.031 [0.028, 0.035] | 0.015 [0.013, 0.016] |
| a01s1 | 0.157 | 0.025 [0.024, 0.026] | 0.013 [0.012, 0.013] | 0.166 | 0.027 [0.024, 0.029] | 0.016 [0.013, 0.017] |
| a02s0 | 0.260 | 0.038 [0.034, 0.044] | 0.022 [0.022, 0.023] | 0.273 | 0.044 [0.040, 0.048] | 0.027 [0.025, 0.030] |
| a02s1 | 0.229 | 0.045 [0.039, 0.045] | 0.023 [0.022, 0.034] | 0.227 | 0.050 [0.045, 0.053] | 0.030 [0.027, 0.033] |
| a03s0 | 0.289 | 0.025 [0.022, 0.033] | 0.022 [0.011, 0.022] | 0.308 | 0.037 [0.034, 0.041] | 0.027 [0.024, 0.029] |
| a03s1 | 0.267 | 0.033 [0.028, 0.034] | 0.022 [0.022, 0.028] | 0.266 | 0.041 [0.039, 0.043] | 0.030 [0.028, 0.033] |
| a05s0 | 0.828 | 0.158 [0.155, 0.167] | 0.156 [0.150, 0.167] | 0.842 | 0.164 [0.158, 0.169] | 0.158 [0.153, 0.161] |
| a05s1 | 0.878 | 0.200 [0.200, 0.211] | 0.200 [0.189, 0.200] | 0.886 | 0.206 [0.198, 0.211] | 0.201 [0.194, 0.205] |
| ELU_s0_E1 | 0.511 | 0.022 [0.012, 0.023] | 0.000 [0.000, 0.000] | 0.549 | 0.022 [0.020, 0.025] | 0.007 [0.004, 0.008] |
| ELU_s0_E2 | 0.291 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.301 | 0.004 [0.003, 0.005] | 0.001 [0.000, 0.001] |
| ELU_s1_E1 | 0.675 | 0.013 [0.012, 0.023] | 0.000 [0.000, 0.006] | 0.736 | 0.020 [0.017, 0.023] | 0.007 [0.005, 0.009] |
| ELU_s1_E2 | 0.389 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.398 | 0.003 [0.002, 0.004] | 0.001 [0.000, 0.001] |

| condition | max N: actual | strat | flat | #(N ≥ 10): actual | strat | flat | excess (F_pool−1)/μ_pool: actual | strat | flat | targets μ=0 per surrogate: strat | flat | targets 0/1-only series: actual | strat | flat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 6 | 2 [1, 2] | 1 [1, 2] | 0 | 0 [0, 0] | 0 [0, 0] | 3.898 | 2.821 [-0.986, 8.626] | -0.960 [-0.966, 24.631] | 51.5 | 76.5 | 0.28 | 0.46 [0.41, 0.56] | 0.24 [0.18, 0.27] |
| a003s1 | 7 | 2 [1, 2] | 1 [1, 2] | 0 | 0 [0, 0] | 0 [0, 0] | 5.065 | 4.967 [-0.981, 16.015] | -0.946 [-0.959, 34.514] | 61.5 | 82.4 | 0.30 | 0.38 [0.29, 0.43] | 0.17 [0.13, 0.24] |
| a01s0 | 7 | 3 [3, 4] | 2 [2, 3] | 0 | 0 [0, 0] | 0 [0, 0] | 3.397 | 5.291 [3.055, 7.161] | 4.584 [1.777, 9.925] | 10.2 | 31.7 | 0.10 | 0.74 [0.64, 0.77] | 0.64 [0.60, 0.72] |
| a01s1 | 6 | 3 [2, 3] | 2 [2, 3] | 0 | 0 [0, 0] | 0 [0, 0] | 3.375 | 4.210 [2.536, 6.523] | 4.669 [0.767, 14.123] | 14.4 | 30.3 | 0.12 | 0.73 [0.67, 0.80] | 0.67 [0.57, 0.70] |
| a02s0 | 11 | 4 [3, 5] | 3 [3, 4] | 2 | 0 [0, 0] | 0 [0, 0] | 3.792 | 8.703 [7.026, 10.839] | 9.162 [6.268, 13.408] | 8.4 | 15.8 | 0.02 | 0.51 [0.43, 0.58] | 0.61 [0.57, 0.70] |
| a02s1 | 8 | 4 [3, 6] | 3 [3, 4] | 0 | 0 [0, 0] | 0 [0, 0] | 3.132 | 8.813 [7.362, 10.766] | 8.540 [6.971, 11.084] | 5.5 | 12.4 | 0.04 | 0.47 [0.43, 0.57] | 0.61 [0.56, 0.71] |
| a03s0 | 14 | 5 [4, 6] | 4 [3, 6] | 3 | 0 [0, 0] | 0 [0, 0] | 3.140 | 17.385 [14.739, 21.241] | 16.694 [13.122, 22.367] | 12.8 | 20.9 | 0.01 | 0.51 [0.46, 0.56] | 0.53 [0.43, 0.61] |
| a03s1 | 9 | 6 [5, 7] | 4 [3, 6] | 0 | 0 [0, 0] | 0 [0, 0] | 2.832 | 17.760 [14.510, 20.183] | 16.144 [12.877, 22.851] | 8.4 | 16.2 | 0.00 | 0.51 [0.43, 0.57] | 0.54 [0.47, 0.59] |
| a05s0 | 10 | 4 [4, 7] | 4 [3, 5] | 1 | 0 [0, 0] | 0 [0, 0] | 0.464 | 0.027 [-0.143, 0.214] | -0.102 [-0.248, 0.096] | 0.0 | 0.0 | 0.00 | 0.44 [0.37, 0.51] | 0.47 [0.41, 0.55] |
| a05s1 | 10 | 4 [3, 6] | 4 [4, 6] | 1 | 0 [0, 0] | 0 [0, 0] | 0.496 | 0.111 [-0.020, 0.255] | 0.107 [-0.000, 0.225] | 0.0 | 0.0 | 0.00 | 0.17 [0.12, 0.23] | 0.18 [0.10, 0.20] |
| ELU_s0_E1 | 18 | 3 [3, 5] | 2 [1, 2] | 32 | 0 [0, 0] | 0 [0, 0] | 4.776 | 13.623 [8.776, 18.559] | 10.016 [-0.983, 23.118] | 23.6 | 60.8 | 0.01 | 0.56 [0.51, 0.64] | 0.38 [0.30, 0.44] |
| ELU_s0_E2 | 11 | 2 [1, 2] | 1 [1, 1] | 1 | 0 [0, 0] | 0 [0, 0] | 4.051 | 15.353 [-0.967, 27.432] | -0.845 [-0.910, 8.094] | 71.8 | 93.4 | 0.04 | 0.27 [0.21, 0.33] | 0.07 [0.03, 0.10] |
| ELU_s1_E1 | 28 | 3 [3, 4] | 2 [1, 3] | 80 | 0 [0, 0] | 0 [0, 0] | 5.256 | 13.582 [8.008, 20.324] | 9.734 [-0.981, 22.394] | 25.1 | 56.8 | 0.00 | 0.58 [0.50, 0.68] | 0.40 [0.34, 0.50] |
| ELU_s1_E2 | 14 | 2 [1, 3] | 1 [1, 1] | 10 | 0 [0, 0] | 0 [0, 0] | 4.681 | 20.847 [-0.959, 64.000] | -0.817 [-0.901, 10.036] | 76.8 | 94.2 | 0.01 | 0.22 [0.17, 0.27] | 0.06 [0.03, 0.10] |

## 3b. B1 — C_i (threshold pool): actual vs strat vs flat (median [5%, 95%] over 20 surrogates)

| condition | F_i median: actual | strat | flat | pooled Fano: actual | strat | flat |
|---|---|---|---|---|---|---|
| a003s0 | 1.345 | 0.982 [0.973, 0.984] | 0.987 [0.985, 1.000] | 1.528 | 1.016 [0.986, 1.055] | 0.995 [0.978, 1.024] |
| a003s1 | 1.326 | 0.984 [0.982, 0.985] | 1.000 [0.986, 1.000] | 1.522 | 1.016 [0.996, 1.041] | 0.998 [0.982, 1.027] |
| a01s0 | 1.354 | 1.035 [0.988, 1.072] | 0.956 [0.925, 0.988] | 1.472 | 1.070 [1.047, 1.105] | 1.002 [0.985, 1.026] |
| a01s1 | 1.348 | 1.005 [0.959, 1.028] | 0.947 [0.928, 0.994] | 1.487 | 1.041 [1.018, 1.066] | 1.003 [0.989, 1.029] |
| a02s0 | 1.677 | 1.097 [1.079, 1.134] | 0.983 [0.955, 1.039] | 1.791 | 1.124 [1.102, 1.148] | 0.995 [0.977, 1.038] |
| a02s1 | 1.446 | 1.095 [1.063, 1.150] | 0.987 [0.958, 1.009] | 1.521 | 1.133 [1.106, 1.162] | 1.000 [0.972, 1.023] |
| a03s0 | 1.803 | 1.123 [1.094, 1.152] | 0.974 [0.962, 0.991] | 1.936 | 1.144 [1.112, 1.173] | 0.985 [0.972, 0.999] |
| a03s1 | 1.573 | 1.124 [1.088, 1.162] | 0.986 [0.950, 1.005] | 1.637 | 1.136 [1.122, 1.158] | 0.988 [0.967, 0.999] |
| a05s0 | 1.743 | 1.086 [1.070, 1.116] | 0.998 [0.961, 1.018] | 1.783 | 1.092 [1.070, 1.120] | 0.999 [0.972, 1.016] |
| a05s1 | 1.612 | 1.087 [1.055, 1.118] | 0.971 [0.951, 1.028] | 1.644 | 1.096 [1.070, 1.106] | 0.977 [0.955, 1.011] |
| ELU_s0_E1 | 3.489 | 1.120 [1.078, 1.153] | 1.018 [0.996, 1.096] | 3.904 | 1.200 [1.162, 1.231] | 1.075 [1.043, 1.116] |
| ELU_s0_E2 | 3.211 | 1.120 [1.084, 1.151] | 1.043 [0.990, 1.078] | 3.623 | 1.177 [1.151, 1.223] | 1.088 [1.056, 1.119] |
| ELU_s1_E1 | 4.501 | 1.105 [1.067, 1.136] | 1.009 [0.954, 1.063] | 5.071 | 1.189 [1.154, 1.228] | 1.068 [1.042, 1.092] |
| ELU_s1_E2 | 3.764 | 1.105 [1.079, 1.148] | 1.015 [0.949, 1.072] | 4.194 | 1.165 [1.133, 1.212] | 1.081 [1.049, 1.111] |

| condition | μ_i median: actual | strat | flat | pooled mean: actual | strat | flat |
|---|---|---|---|---|---|---|
| a003s0 | 0.287 | 0.038 [0.034, 0.044] | 0.019 [0.016, 0.027] | 0.335 | 0.041 [0.038, 0.045] | 0.023 [0.021, 0.027] |
| a003s1 | 0.222 | 0.030 [0.029, 0.033] | 0.016 [0.015, 0.017] | 0.264 | 0.033 [0.031, 0.036] | 0.019 [0.017, 0.021] |
| a01s0 | 0.479 | 0.165 [0.158, 0.172] | 0.142 [0.133, 0.146] | 0.493 | 0.166 [0.162, 0.173] | 0.141 [0.135, 0.147] |
| a01s1 | 0.447 | 0.152 [0.145, 0.163] | 0.140 [0.132, 0.144] | 0.467 | 0.156 [0.147, 0.163] | 0.143 [0.135, 0.147] |
| a02s0 | 1.097 | 0.493 [0.478, 0.506] | 0.611 [0.596, 0.621] | 1.102 | 0.493 [0.485, 0.504] | 0.613 [0.602, 0.625] |
| a02s1 | 0.966 | 0.489 [0.478, 0.503] | 0.601 [0.586, 0.611] | 0.968 | 0.495 [0.485, 0.506] | 0.602 [0.593, 0.610] |
| a03s0 | 1.944 | 1.203 [1.188, 1.223] | 1.802 [1.760, 1.834] | 2.027 | 1.209 [1.197, 1.224] | 1.810 [1.772, 1.834] |
| a03s1 | 1.861 | 1.212 [1.190, 1.233] | 1.783 [1.766, 1.811] | 1.874 | 1.217 [1.203, 1.233] | 1.780 [1.764, 1.798] |
| a05s0 | 6.867 | 6.411 [6.361, 6.456] | 9.494 [9.416, 9.579] | 6.890 | 6.423 [6.383, 6.444] | 9.496 [9.436, 9.557] |
| a05s1 | 6.850 | 6.381 [6.333, 6.429] | 9.594 [9.566, 9.684] | 6.914 | 6.384 [6.356, 6.421] | 9.602 [9.570, 9.690] |
| ELU_s0_E1 | 1.953 | 0.160 [0.152, 0.171] | 0.126 [0.117, 0.134] | 1.968 | 0.166 [0.158, 0.169] | 0.128 [0.123, 0.132] |
| ELU_s0_E2 | 2.102 | 0.231 [0.222, 0.243] | 0.127 [0.123, 0.136] | 2.084 | 0.235 [0.228, 0.242] | 0.129 [0.121, 0.135] |
| ELU_s1_E1 | 2.221 | 0.143 [0.135, 0.151] | 0.116 [0.113, 0.125] | 2.215 | 0.149 [0.143, 0.158] | 0.120 [0.115, 0.125] |
| ELU_s1_E2 | 2.199 | 0.199 [0.191, 0.207] | 0.118 [0.112, 0.124] | 2.201 | 0.202 [0.196, 0.210] | 0.119 [0.113, 0.124] |

| condition | max C: actual | strat | flat | #(C ≥ 10): actual | strat | flat | excess (F_pool−1)/μ_pool: actual | strat | flat | targets μ=0 per surrogate: strat | flat | targets 0/1-only series: actual | strat | flat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 6 | 2 [2, 3] | 2 [1, 2] | 0 | 0 [0, 0] | 0 [0, 0] | 1.577 | 0.402 [-0.371, 1.361] | -0.204 [-0.993, 1.150] | 7.5 | 21.1 | 0.08 | 0.85 [0.81, 0.90] | 0.76 [0.71, 0.83] |
| a003s1 | 7 | 2 [2, 2] | 2 [1, 2] | 0 | 0 [0, 0] | 0 [0, 0] | 1.976 | 0.480 [-0.124, 1.234] | -0.121 [-0.992, 1.330] | 13.6 | 30.4 | 0.12 | 0.81 [0.75, 0.86] | 0.69 [0.60, 0.74] |
| a01s0 | 9 | 4 [3, 4] | 3 [3, 4] | 0 | 0 [0, 0] | 0 [0, 0] | 0.958 | 0.430 [0.282, 0.631] | 0.017 [-0.112, 0.188] | 0.1 | 0.0 | 0.00 | 0.29 [0.22, 0.33] | 0.46 [0.41, 0.52] |
| a01s1 | 10 | 3 [3, 4] | 3 [3, 4] | 1 | 0 [0, 0] | 0 [0, 0] | 1.042 | 0.272 [0.113, 0.429] | 0.018 [-0.077, 0.199] | 0.1 | 0.0 | 0.00 | 0.35 [0.31, 0.40] | 0.47 [0.38, 0.52] |
| a02s0 | 12 | 6 [5, 7] | 5 [5, 6] | 10 | 0 [0, 0] | 0 [0, 0] | 0.718 | 0.251 [0.209, 0.300] | -0.009 [-0.038, 0.062] | 0.0 | 0.0 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| a02s1 | 11 | 6 [5, 7] | 5 [5, 6] | 5 | 0 [0, 0] | 0 [0, 0] | 0.539 | 0.270 [0.215, 0.321] | -0.000 [-0.048, 0.039] | 0.0 | 0.0 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| a03s0 | 16 | 8 [8, 10] | 9 [8, 9] | 67 | 0 [0, 1] | 0 [0, 0] | 0.462 | 0.118 [0.094, 0.143] | -0.008 [-0.016, -0.000] | 0.0 | 0.0 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| a03s1 | 14 | 8 [7, 10] | 8 [8, 9] | 17 | 0 [0, 1] | 0 [0, 0] | 0.340 | 0.112 [0.100, 0.129] | -0.007 [-0.019, -0.000] | 0.0 | 0.0 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| a05s0 | 27 | 19 [18, 20] | 23 [22, 25] | 1818 | 1123 [1100, 1168] | 4287 [4189, 4407] | 0.114 | 0.014 [0.011, 0.019] | -0.000 [-0.003, 0.002] | 0.0 | 0.0 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| a05s1 | 27 | 19 [18, 21] | 23 [21, 25] | 1802 | 1088 [1069, 1127] | 4398 [4341, 4513] | 0.093 | 0.015 [0.011, 0.017] | -0.002 [-0.005, 0.001] | 0.0 | 0.0 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| ELU_s0_E1 | 22 | 5 [4, 6] | 4 [3, 4] | 259 | 0 [0, 0] | 0 [0, 0] | 1.476 | 1.226 [0.999, 1.402] | 0.589 [0.340, 0.950] | 0.0 | 0.0 | 0.00 | 0.20 [0.14, 0.24] | 0.39 [0.29, 0.44] |
| ELU_s0_E2 | 24 | 5 [4, 6] | 4 [3, 5] | 231 | 0 [0, 0] | 0 [0, 0] | 1.259 | 0.750 [0.649, 0.954] | 0.672 [0.451, 0.948] | 0.0 | 0.0 | 0.00 | 0.07 [0.04, 0.11] | 0.36 [0.32, 0.45] |
| ELU_s1_E1 | 33 | 5 [4, 6] | 3 [3, 4] | 393 | 0 [0, 0] | 0 [0, 0] | 1.838 | 1.239 [1.037, 1.462] | 0.579 [0.361, 0.775] | 0.0 | 0.0 | 0.00 | 0.23 [0.16, 0.28] | 0.42 [0.37, 0.51] |
| ELU_s1_E2 | 26 | 5 [4, 7] | 4 [3, 4] | 308 | 0 [0, 0] | 0 [0, 0] | 1.451 | 0.823 [0.672, 1.013] | 0.673 [0.409, 0.935] | 0.0 | 0.0 | 0.00 | 0.11 [0.08, 0.17] | 0.45 [0.35, 0.54] |

Note [恒等式]: a 0/1 series of length n with m ones has F = n/(n−1)·(1 − m/n), i.e. ≈ 1 for sparse series; the last columns give the share of targets whose series is 0/1-only (μ > 0), so surrogate F_i medians near 1.000 are this arithmetic. The excess index (F_pool − 1)/μ_pool is listed because the surrogate means differ from the actual mean.

## 4. B2 — per-target position of the actual F_i (and μ_i) in the 20 surrogate values of the same target

inside = actual within [5%, 95%] of the target's 20 surrogate values; above = actual > 95%; n = targets with finite actual F_i and finite surrogate quantiles.

**N_i**

| condition | strat F: n | inside 90% | above q95 | below q05 | flat F: n | inside 90% | above q95 | below q05 | strat μ_i: inside | above | flat μ_i: inside | above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 99 | 0.15 | 0.60 | 0.25 | 98 | 0.04 | 0.69 | 0.27 | 0.05 | 0.95 | 0.03 | 0.97 |
| a003s1 | 99 | 0.19 | 0.58 | 0.23 | 99 | 0.05 | 0.69 | 0.26 | 0.09 | 0.91 | 0.05 | 0.95 |
| a01s0 | 100 | 0.51 | 0.41 | 0.08 | 100 | 0.26 | 0.65 | 0.09 | 0.03 | 0.97 | 0.01 | 0.99 |
| a01s1 | 100 | 0.44 | 0.45 | 0.11 | 100 | 0.30 | 0.58 | 0.12 | 0.05 | 0.95 | 0.03 | 0.97 |
| a02s0 | 100 | 0.55 | 0.43 | 0.02 | 100 | 0.51 | 0.47 | 0.02 | 0.00 | 1.00 | 0.00 | 1.00 |
| a02s1 | 100 | 0.68 | 0.27 | 0.05 | 100 | 0.57 | 0.38 | 0.05 | 0.01 | 0.99 | 0.00 | 1.00 |
| a03s0 | 100 | 0.56 | 0.41 | 0.03 | 100 | 0.46 | 0.51 | 0.03 | 0.00 | 1.00 | 0.00 | 1.00 |
| a03s1 | 100 | 0.78 | 0.20 | 0.02 | 100 | 0.60 | 0.39 | 0.01 | 0.00 | 1.00 | 0.00 | 1.00 |
| a05s0 | 100 | 0.35 | 0.65 | 0.00 | 100 | 0.29 | 0.71 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| a05s1 | 100 | 0.34 | 0.66 | 0.00 | 100 | 0.35 | 0.65 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| ELU_s0_E1 | 100 | 0.08 | 0.91 | 0.01 | 99 | 0.01 | 0.98 | 0.01 | 0.00 | 1.00 | 0.00 | 1.00 |
| ELU_s0_E2 | 95 | 0.02 | 0.93 | 0.05 | 65 | 0.00 | 0.94 | 0.06 | 0.00 | 1.00 | 0.00 | 1.00 |
| ELU_s1_E1 | 100 | 0.06 | 0.94 | 0.00 | 100 | 0.01 | 0.99 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| ELU_s1_E2 | 97 | 0.05 | 0.92 | 0.03 | 70 | 0.00 | 0.96 | 0.04 | 0.00 | 1.00 | 0.00 | 1.00 |

**C_i**

| condition | strat F: n | inside 90% | above q95 | below q05 | flat F: n | inside 90% | above q95 | below q05 | strat μ_i: inside | above | flat μ_i: inside | above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 100 | 0.20 | 0.70 | 0.10 | 100 | 0.06 | 0.80 | 0.14 | 0.03 | 0.97 | 0.01 | 0.99 |
| a003s1 | 100 | 0.19 | 0.70 | 0.11 | 100 | 0.09 | 0.78 | 0.13 | 0.04 | 0.96 | 0.01 | 0.99 |
| a01s0 | 100 | 0.52 | 0.48 | 0.00 | 100 | 0.33 | 0.67 | 0.00 | 0.03 | 0.97 | 0.01 | 0.99 |
| a01s1 | 100 | 0.44 | 0.55 | 0.01 | 100 | 0.32 | 0.66 | 0.02 | 0.02 | 0.98 | 0.03 | 0.97 |
| a02s0 | 100 | 0.24 | 0.76 | 0.00 | 100 | 0.13 | 0.87 | 0.00 | 0.00 | 1.00 | 0.06 | 0.94 |
| a02s1 | 100 | 0.44 | 0.55 | 0.01 | 100 | 0.19 | 0.81 | 0.00 | 0.03 | 0.97 | 0.06 | 0.94 |
| a03s0 | 100 | 0.05 | 0.95 | 0.00 | 100 | 0.02 | 0.98 | 0.00 | 0.02 | 0.98 | 0.48 | 0.44 |
| a03s1 | 100 | 0.25 | 0.75 | 0.00 | 100 | 0.09 | 0.91 | 0.00 | 0.02 | 0.98 | 0.58 | 0.31 |
| a05s0 | 100 | 0.07 | 0.93 | 0.00 | 100 | 0.04 | 0.96 | 0.00 | 0.39 | 0.58 | 0.00 | 0.00 |
| a05s1 | 100 | 0.15 | 0.85 | 0.00 | 100 | 0.04 | 0.96 | 0.00 | 0.35 | 0.64 | 0.00 | 0.00 |
| ELU_s0_E1 | 100 | 0.01 | 0.99 | 0.00 | 100 | 0.01 | 0.99 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| ELU_s0_E2 | 100 | 0.00 | 1.00 | 0.00 | 100 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| ELU_s1_E1 | 100 | 0.02 | 0.98 | 0.00 | 100 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| ELU_s1_E2 | 100 | 0.00 | 1.00 | 0.00 | 100 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |

## 5. B3 — common-cause check: actual series vs the surrogate-mean series N̄_i(T) = mean over the 20 surrogates at the same T

Spearman (average ranks) and Pearson over all (target, alive phase-2 T); within-target = median over targets of the correlation over T; var ratio = median over targets of Var_T(N̄)/Var_T(actual) (ddof 1) [q25, q75]; noise-corrected = (Var_T(N̄) − mean_T Var_r(N^(r))/20)/Var_T(actual), median.

**N_i**

| condition | strat: Spearman (i,T) | Pearson (i,T) | within-target Spearman | var ratio [q25, q75] | var ratio noise-corr. | flat: Spearman (i,T) | within-target Spearman | var ratio | noise-corr. |
|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.240 | 0.234 | 0.228 | 0.0044 [0.0023, 0.0106] | 0.0018 | 0.192 | 0.167 | 0.0013 | 0.0000 |
| a003s1 | 0.204 | 0.185 | 0.219 | 0.0036 [0.0018, 0.0092] | 0.0010 | 0.136 | 0.121 | 0.0011 | -0.0000 |
| a01s0 | 0.359 | 0.353 | 0.359 | 0.0236 [0.0146, 0.0418] | 0.0174 | 0.343 | 0.356 | 0.0066 | 0.0039 |
| a01s1 | 0.340 | 0.313 | 0.364 | 0.0207 [0.0125, 0.0356] | 0.0145 | 0.297 | 0.306 | 0.0086 | 0.0051 |
| a02s0 | 0.440 | 0.457 | 0.438 | 0.0333 [0.0157, 0.0600] | 0.0293 | 0.418 | 0.406 | 0.0149 | 0.0115 |
| a02s1 | 0.443 | 0.476 | 0.427 | 0.0601 [0.0332, 0.1010] | 0.0553 | 0.407 | 0.409 | 0.0235 | 0.0193 |
| a03s0 | 0.399 | 0.499 | 0.406 | 0.0206 [0.0085, 0.0618] | 0.0180 | 0.370 | 0.369 | 0.0121 | 0.0104 |
| a03s1 | 0.405 | 0.481 | 0.408 | 0.0368 [0.0172, 0.0867] | 0.0326 | 0.373 | 0.386 | 0.0235 | 0.0208 |
| a05s0 | 0.340 | 0.324 | 0.344 | 0.0578 [0.0426, 0.0670] | 0.0531 | 0.334 | 0.333 | 0.0487 | 0.0436 |
| a05s1 | 0.364 | 0.336 | 0.367 | 0.0659 [0.0529, 0.0890] | 0.0608 | 0.336 | 0.326 | 0.0662 | 0.0609 |
| ELU_s0_E1 | 0.407 | 0.352 | 0.392 | 0.0038 [0.0012, 0.0079] | 0.0029 | 0.341 | 0.354 | 0.0004 | 0.0002 |
| ELU_s0_E2 | 0.206 | 0.178 | 0.218 | 0.0005 [0.0002, 0.0014] | 0.0002 | 0.117 | 0.194 | 0.0000 | 0.0000 |
| ELU_s1_E1 | 0.392 | 0.335 | 0.414 | 0.0018 [0.0007, 0.0041] | 0.0013 | 0.330 | 0.346 | 0.0002 | 0.0001 |
| ELU_s1_E2 | 0.165 | 0.140 | 0.177 | 0.0002 [0.0001, 0.0006] | 0.0001 | 0.111 | 0.158 | 0.0000 | 0.0000 |

**C_i**

| condition | strat: Spearman (i,T) | Pearson (i,T) | within-target Spearman | var ratio [q25, q75] | var ratio noise-corr. | flat: Spearman (i,T) | within-target Spearman | var ratio | noise-corr. |
|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.129 | 0.111 | 0.127 | 0.0082 [0.0047, 0.0147] | 0.0029 | 0.042 | 0.027 | 0.0030 | 0.0001 |
| a003s1 | 0.113 | 0.107 | 0.111 | 0.0073 [0.0040, 0.0121] | 0.0019 | 0.016 | 0.021 | 0.0028 | 0.0001 |
| a01s0 | 0.277 | 0.275 | 0.282 | 0.0377 [0.0260, 0.0537] | 0.0248 | 0.073 | 0.075 | 0.0141 | 0.0030 |
| a01s1 | 0.226 | 0.216 | 0.215 | 0.0297 [0.0191, 0.0370] | 0.0158 | 0.062 | 0.047 | 0.0154 | 0.0042 |
| a02s0 | 0.372 | 0.351 | 0.376 | 0.0608 [0.0424, 0.0824] | 0.0456 | 0.085 | 0.093 | 0.0314 | 0.0136 |
| a02s1 | 0.347 | 0.353 | 0.359 | 0.0747 [0.0609, 0.1015] | 0.0590 | 0.114 | 0.115 | 0.0362 | 0.0163 |
| a03s0 | 0.415 | 0.412 | 0.424 | 0.0826 [0.0682, 0.1132] | 0.0694 | 0.118 | 0.129 | 0.0475 | 0.0255 |
| a03s1 | 0.406 | 0.420 | 0.402 | 0.1070 [0.0919, 0.1340] | 0.0886 | 0.115 | 0.120 | 0.0633 | 0.0333 |
| a05s0 | 0.420 | 0.436 | 0.425 | 0.1716 [0.1448, 0.2013] | 0.1490 | 0.247 | 0.243 | 0.1736 | 0.1391 |
| a05s1 | 0.426 | 0.449 | 0.427 | 0.1885 [0.1674, 0.2232] | 0.1655 | 0.229 | 0.237 | 0.1854 | 0.1477 |
| ELU_s0_E1 | 0.326 | 0.216 | 0.339 | 0.0054 [0.0033, 0.0087] | 0.0042 | -0.028 | -0.012 | 0.0025 | 0.0016 |
| ELU_s0_E2 | 0.246 | 0.172 | 0.250 | 0.0076 [0.0046, 0.0118] | 0.0058 | -0.024 | -0.028 | 0.0029 | 0.0018 |
| ELU_s1_E1 | 0.293 | 0.187 | 0.299 | 0.0033 [0.0018, 0.0054] | 0.0025 | -0.058 | -0.050 | 0.0016 | 0.0010 |
| ELU_s1_E2 | 0.165 | 0.095 | 0.191 | 0.0047 [0.0031, 0.0080] | 0.0036 | -0.030 | -0.049 | 0.0019 | 0.0012 |

## 6. B5 — actual brightness profile preserved by strat (phase 2 pooled)

occupancy(d) = open gates of alive units on decile-d inputs / (120 × alive units), pooled over T 11..100; decile 0 = smallest K. T_i share(d) = share of the alive targets' threshold inputs (3 per target-state) that lie in decile d.

| condition | overall | occ d0 | occ d1 | occ d2 | occ d3 | occ d4 | occ d5 | occ d6 | occ d7 | occ d8 | occ d9 | d9/d0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.0061 | 0.0157 | 0.0104 | 0.0075 | 0.0064 | 0.0063 | 0.0052 | 0.0032 | 0.0029 | 0.0021 | 0.0008 | 0.052 |
| a003s1 | 0.0056 | 0.0136 | 0.0099 | 0.0080 | 0.0068 | 0.0052 | 0.0043 | 0.0030 | 0.0026 | 0.0016 | 0.0006 | 0.042 |
| a01s0 | 0.0141 | 0.0307 | 0.0229 | 0.0180 | 0.0169 | 0.0143 | 0.0118 | 0.0089 | 0.0080 | 0.0062 | 0.0035 | 0.115 |
| a01s1 | 0.0140 | 0.0257 | 0.0210 | 0.0188 | 0.0172 | 0.0146 | 0.0128 | 0.0105 | 0.0091 | 0.0066 | 0.0032 | 0.125 |
| a02s0 | 0.0322 | 0.0568 | 0.0460 | 0.0393 | 0.0378 | 0.0344 | 0.0296 | 0.0254 | 0.0228 | 0.0187 | 0.0112 | 0.197 |
| a02s1 | 0.0314 | 0.0547 | 0.0451 | 0.0392 | 0.0378 | 0.0327 | 0.0297 | 0.0243 | 0.0219 | 0.0179 | 0.0103 | 0.188 |
| a03s0 | 0.0619 | 0.0975 | 0.0809 | 0.0744 | 0.0700 | 0.0653 | 0.0600 | 0.0520 | 0.0488 | 0.0411 | 0.0293 | 0.300 |
| a03s1 | 0.0612 | 0.0903 | 0.0797 | 0.0738 | 0.0722 | 0.0654 | 0.0597 | 0.0534 | 0.0490 | 0.0403 | 0.0277 | 0.307 |
| a05s0 | 0.1737 | 0.2157 | 0.1976 | 0.1912 | 0.1868 | 0.1824 | 0.1753 | 0.1651 | 0.1576 | 0.1453 | 0.1196 | 0.554 |
| a05s1 | 0.1744 | 0.2083 | 0.2019 | 0.1944 | 0.1937 | 0.1842 | 0.1792 | 0.1666 | 0.1573 | 0.1444 | 0.1143 | 0.549 |
| ELU_s0_E1 | 0.0159 | 0.0412 | 0.0275 | 0.0217 | 0.0172 | 0.0191 | 0.0116 | 0.0076 | 0.0067 | 0.0050 | 0.0018 | 0.043 |
| ELU_s0_E2 | 0.0159 | 0.0412 | 0.0275 | 0.0217 | 0.0172 | 0.0191 | 0.0116 | 0.0076 | 0.0067 | 0.0050 | 0.0018 | 0.043 |
| ELU_s1_E1 | 0.0154 | 0.0342 | 0.0277 | 0.0236 | 0.0197 | 0.0144 | 0.0111 | 0.0089 | 0.0079 | 0.0048 | 0.0016 | 0.046 |
| ELU_s1_E2 | 0.0154 | 0.0342 | 0.0277 | 0.0236 | 0.0197 | 0.0144 | 0.0111 | 0.0089 | 0.0079 | 0.0048 | 0.0016 | 0.046 |

| condition | T_i d0 | T_i d1 | T_i d2 | T_i d3 | T_i d4 | T_i d5 | T_i d6 | T_i d7 | T_i d8 | T_i d9 |
|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.209 | 0.151 | 0.114 | 0.102 | 0.107 | 0.091 | 0.063 | 0.073 | 0.060 | 0.030 |
| a003s1 | 0.196 | 0.149 | 0.128 | 0.125 | 0.103 | 0.088 | 0.071 | 0.070 | 0.046 | 0.023 |
| a01s0 | 0.107 | 0.102 | 0.093 | 0.104 | 0.103 | 0.101 | 0.095 | 0.103 | 0.106 | 0.086 |
| a01s1 | 0.091 | 0.091 | 0.097 | 0.107 | 0.104 | 0.106 | 0.103 | 0.113 | 0.110 | 0.078 |
| a02s0 | 0.031 | 0.047 | 0.049 | 0.069 | 0.081 | 0.093 | 0.113 | 0.144 | 0.186 | 0.185 |
| a02s1 | 0.032 | 0.043 | 0.054 | 0.074 | 0.082 | 0.101 | 0.110 | 0.147 | 0.183 | 0.173 |
| a03s0 | 0.007 | 0.015 | 0.020 | 0.033 | 0.048 | 0.066 | 0.094 | 0.148 | 0.226 | 0.344 |
| a03s1 | 0.006 | 0.012 | 0.019 | 0.036 | 0.048 | 0.066 | 0.097 | 0.149 | 0.242 | 0.326 |
| a05s0 | 0.000 | 0.000 | 0.000 | 0.001 | 0.003 | 0.008 | 0.020 | 0.055 | 0.185 | 0.727 |
| a05s1 | 0.000 | 0.000 | 0.000 | 0.001 | 0.004 | 0.009 | 0.022 | 0.059 | 0.203 | 0.702 |
| ELU_s0_E1 | 0.112 | 0.100 | 0.109 | 0.103 | 0.153 | 0.109 | 0.090 | 0.094 | 0.086 | 0.043 |
| ELU_s0_E2 | 0.225 | 0.155 | 0.136 | 0.106 | 0.130 | 0.077 | 0.056 | 0.052 | 0.043 | 0.019 |
| ELU_s1_E1 | 0.102 | 0.108 | 0.113 | 0.128 | 0.110 | 0.105 | 0.105 | 0.107 | 0.085 | 0.038 |
| ELU_s1_E2 | 0.196 | 0.160 | 0.146 | 0.136 | 0.098 | 0.080 | 0.068 | 0.058 | 0.042 | 0.017 |

K-decile mean K, decile 0 … decile 9 (the 1,200-input subset is identical within a seed in the files checked: LR_ref, ELU_ref, LRa0.03, LRa0.5): a01s0: 19.5…53.9; a01s1: 19.7…55.7

## 7. Files

- per-condition arrays for the coordinator: `/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b/b_arrays_<tag>.npz`, keys N, C (int16, −1 = not alive), Nstrat_mean, Cstrat_mean, Nstrat_draws (int16 [r, T, i]), Nflat_mean, Cflat_mean (float, −1 = not alive), T_list (11..100); rows T 11..100, columns all 100 units; written by `b_arrays.py`.
- full surrogate arrays (N and C, strat and flat, 20 draws each) + K + decile labels: `.../agent_b/b_surr_<tag>.npz`. Logs: `.../agent_b/logs/b_<tag>.log`.
- `<tag>` ∈ a003s0, a003s1, a01s0, a01s1, a02s0, a02s1, a03s0, a03s1, a05s0, a05s1, ELU_s0_E1, ELU_s0_E2, ELU_s1_E1, ELU_s1_E2.

## 8. iso and iso+common surrogates (coordinator follow-ups)

Same box, definitions, 14 conditions, targets and statistics as §1–6; script `analysis/xbar_arrangement_1014/b_iso_null.py` (mode `iso` / `isoc`), tables by `b_report_iso.py`. Per phase-2 state and alive unit k: x̄ = mean_n Xt_n (785), K̄ = mean K = x̄·x̄, m_k = mean_n z_kn, u⊥_kn = z_kn − m_k K_n/K̄, s⊥_k = sd_n(u⊥_kn) (ddof 0).

- **iso**: g ~ N(0, I_785), g ← g − x̄(x̄·g)/K̄, u = Xt·g, u rescaled to sd_n(u) = s⊥_k, z^iso_kn = m_k K_n/K̄ + u_n with the actual m_k; gates 1[z^iso > 0]. Seeds `default_rng(2000 + r)`.
- **iso+common (isoc)**: ū⊥_n = mean over alive k of u⊥_kn; the isotropic u (fresh g, same projection) is rescaled to sd_n(u) = sqrt(max(s⊥_k² − sd_n(ū⊥)², 0.25 s⊥_k²)); z^isoc_kn = m_k K_n/K̄ + ū⊥_n + u_n. Seeds `default_rng(3000 + r)`.
- Both: one g per (unit, state, draw), R = 20, drawn state by state T = 11..100 for all 100 columns; candidate eligibility = the unit's actual alive status; the target's own T_i, O_i, k_i stay actual. Counts reproduce cobirth_seeds in all 28 runs (14 iso + 14 isoc: MATCH). The iso rerun of ELU_s1_E2 after adding the isoc mode gave bit-identical draws.

Criterion: the null reproduces a pooled statistic when the actual value lies in the [5%, 95%] range of its 20 draws. 'ratio' = (median over draws of the pooled mean) ÷ (actual pooled mean).

### 8.0 Verdicts

- **iso reproduces μ of N**: ratio leaky 0.82–1.32, ELU 0.21–0.24; pooled mean in band 0/14 (leaky 0/10, ELU 0/4).
- **iso reproduces μ of C**: ratio leaky 0.89–1.15, ELU 0.24–0.31; pooled mean in band 2/14 (leaky 2/10, ELU 0/4).
- **iso reproduces pooled Fano of N**: 5/14 in band (leaky 5/10, ELU 0/4); actual leaky 1.39–2.03, ELU 2.22–4.87; iso leaky 1.33–1.96, ELU 1.53–2.11. Median F_i in band 7/14.
- **iso reproduces pooled Fano of C**: 4/14 in band (leaky 4/10, ELU 0/4); actual leaky 1.47–1.94, ELU 3.62–5.07; iso leaky 1.29–1.77, ELU 1.61–1.84. Median F_i in band 5/14.
- **iso+common reproduces μ of N**: ratio leaky 0.75–1.45, ELU 0.97–1.07; in band 4/14 (leaky 1/10, ELU 3/4).
- **iso+common reproduces μ of C**: ratio leaky 0.86–1.12, ELU 0.90–0.96; in band 2/14 (leaky 2/10, ELU 0/4).
- **iso+common reproduces pooled Fano of N**: 4/14 in band (leaky 2/10, ELU 2/4); isoc leaky 1.33–1.94, ELU 2.20–4.92. Median F_i in band 7/14.
- **iso+common reproduces pooled Fano of C**: 6/14 in band (leaky 3/10, ELU 3/4); isoc leaky 1.28–1.84, ELU 3.56–5.29. Median F_i in band 8/14.
- **iso vs strat vs flat vs iso+common**, in-band counts out of 14 (pooled Fano N / pooled Fano C / pooled mean N / pooled mean C / median F_i N / median F_i C): strat 1/0/0/0/0/0; flat 0/0/0/0/0/0; iso 5/4/0/2/7/5; isoc 4/6/4/2/7/8.
- **ratio of pooled means null ÷ actual** (N; C), leaky | ELU: strat N 0.07–0.23 | 0.01–0.04, C 0.12–0.93 | 0.07–0.11; flat N 0.02–0.23 | 0.00–0.01, C 0.07–1.39 | 0.05–0.06; iso N 0.82–1.32 | 0.21–0.24, C 0.89–1.15 | 0.24–0.31; isoc N 0.75–1.45 | 0.97–1.07, C 0.86–1.12 | 0.90–0.96.
- **Does adding the common component close the gap?** median over conditions of |null − actual|/actual, pooled Fano (N; C) and pooled mean (N; C): iso: Fano leaky 0.026; 0.055, ELU 0.440; 0.580 — mean leaky 0.143; 0.076, ELU 0.778; 0.722; isoc: Fano leaky 0.067; 0.071, ELU 0.025; 0.021 — mean leaky 0.147; 0.076, ELU 0.025; 0.068.
- **ELU (B4)**: max N actual 18, 11, 28, 14; iso 8, 6, 9, 7; isoc 18, 11, 28, 18. #N≥10 actual 32, 1, 80, 10; iso 0 [0, 1], 0 [0, 0], 0 [0, 1], 0 [0, 0]; isoc 33 [28, 41], 2 [0, 4], 73 [68, 85], 14 [8, 19].
- **Widths** (realized mean p ÷ actual mean p): iso leaky 1.10–1.36, ELU 0.82–0.84; isoc leaky 1.09–1.20, ELU 0.95–0.96. Common component sd(ū⊥)/mean s⊥ (median over states): leaky 0.107–0.257, ELU 0.343–0.357; variance share leaky 0.011–0.065, ELU 0.116–0.126; residual floor never active (0 in all states).
- **N_i per target (B2) and common cause (B3)**: actual F_i inside the 5–95% band, iso leaky 0.70–0.82, ELU 0.20–0.54; isoc leaky 0.71–0.81, ELU 0.82–0.84 (strat leaky 0.15–0.78, ELU 0.02–0.08). Spearman (i,T) iso leaky 0.32–0.51, ELU 0.40–0.57; isoc leaky 0.33–0.53, ELU 0.55–0.69. Variance share (median var ratio) iso leaky 0.197–0.590, ELU 0.045–0.054; isoc leaky 0.207–0.706, ELU 0.576–0.756.
- **C_i per target (B2) and common cause (B3)**: actual F_i inside the 5–95% band, iso leaky 0.68–0.76, ELU 0.03–0.08; isoc leaky 0.63–0.85, ELU 0.85–0.93 (strat leaky 0.05–0.52, ELU 0.00–0.02). Spearman (i,T) iso leaky 0.33–0.64, ELU 0.53–0.58; isoc leaky 0.33–0.71, ELU 0.79–0.80. Variance share (median var ratio) iso leaky 0.252–0.560, ELU 0.042–0.061; isoc leaky 0.246–0.717, ELU 0.697–0.759.

### 8.1 Realized widths and size of the common component

p = open fraction k/1200, mean over alive units, pooled over phase-2 states (and draws for the surrogates). Unit-level columns: draw r = 0, all alive (unit, state) pairs. Common component: sd ratio = sd_n(ū⊥)/mean_k s⊥_k; var share = sd_n(ū⊥)²/mean_k s⊥_k²; unit R² = median over alive units of corr_n(u⊥_k, ū⊥)²; floor active = share of alive units with s⊥_k² − sd(ū⊥)² < 0.25 s⊥_k² (isoc residual floored). Median [5%, 95%] over the 90 phase-2 states.

| condition | p actual | p iso | iso/act | p isoc | isoc/act | iso unit-level Pearson(p) | iso units with k=0 | sd(ū⊥)/mean s⊥ | var share | unit R² | floor active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.0060 | 0.0079 | 1.308 | 0.0070 | 1.161 | 0.840 | 0.278 | 0.238 [0.159, 0.308] | 0.056 [0.025, 0.093] | 0.056 [0.028, 0.103] | 0.000 [0.000, 0.000] |
| a003s1 | 0.0056 | 0.0075 | 1.355 | 0.0066 | 1.196 | 0.804 | 0.284 | 0.229 [0.161, 0.321] | 0.052 [0.025, 0.102] | 0.054 [0.028, 0.123] | 0.000 [0.000, 0.000] |
| a01s0 | 0.0141 | 0.0182 | 1.292 | 0.0162 | 1.149 | 0.915 | 0.116 | 0.257 [0.174, 0.308] | 0.065 [0.030, 0.093] | 0.071 [0.035, 0.114] | 0.000 [0.000, 0.000] |
| a01s1 | 0.0139 | 0.0184 | 1.317 | 0.0160 | 1.149 | 0.915 | 0.122 | 0.246 [0.150, 0.309] | 0.059 [0.022, 0.093] | 0.063 [0.027, 0.106] | 0.000 [0.000, 0.000] |
| a02s0 | 0.0322 | 0.0382 | 1.186 | 0.0357 | 1.109 | 0.951 | 0.035 | 0.198 [0.155, 0.237] | 0.039 [0.023, 0.055] | 0.050 [0.031, 0.072] | 0.000 [0.000, 0.000] |
| a02s1 | 0.0313 | 0.0366 | 1.168 | 0.0342 | 1.093 | 0.951 | 0.039 | 0.201 [0.141, 0.232] | 0.040 [0.019, 0.053] | 0.046 [0.029, 0.061] | 0.000 [0.000, 0.000] |
| a03s0 | 0.0619 | 0.0707 | 1.141 | 0.0685 | 1.105 | 0.967 | 0.006 | 0.169 [0.140, 0.193] | 0.028 [0.019, 0.037] | 0.038 [0.024, 0.052] | 0.000 [0.000, 0.000] |
| a03s1 | 0.0611 | 0.0703 | 1.150 | 0.0682 | 1.116 | 0.965 | 0.005 | 0.147 [0.125, 0.181] | 0.021 [0.015, 0.032] | 0.030 [0.020, 0.041] | 0.000 [0.000, 0.000] |
| a05s0 | 0.1737 | 0.1922 | 1.106 | 0.1916 | 1.103 | 0.975 | 0.000 | 0.118 [0.094, 0.142] | 0.014 [0.009, 0.020] | 0.019 [0.013, 0.027] | 0.000 [0.000, 0.000] |
| a05s1 | 0.1744 | 0.1918 | 1.100 | 0.1911 | 1.096 | 0.976 | 0.000 | 0.107 [0.089, 0.122] | 0.011 [0.008, 0.015] | 0.017 [0.011, 0.025] | 0.000 [0.000, 0.000] |
| ELU_s0_E1 | 0.0159 | 0.0130 | 0.819 | 0.0153 | 0.963 | 0.882 | 0.139 | 0.357 [0.158, 0.433] | 0.126 [0.024, 0.184] | 0.137 [0.031, 0.207] | 0.000 [0.000, 0.000] |
| ELU_s0_E2 | 0.0159 | 0.0130 | 0.819 | 0.0153 | 0.963 | 0.882 | 0.139 | 0.357 [0.158, 0.433] | 0.126 [0.024, 0.184] | 0.137 [0.031, 0.207] | 0.000 [0.000, 0.000] |
| ELU_s1_E1 | 0.0154 | 0.0129 | 0.842 | 0.0147 | 0.954 | 0.869 | 0.158 | 0.343 [0.163, 0.403] | 0.116 [0.026, 0.158] | 0.129 [0.033, 0.192] | 0.000 [0.000, 0.000] |
| ELU_s1_E2 | 0.0154 | 0.0129 | 0.842 | 0.0147 | 0.954 | 0.869 | 0.158 | 0.343 [0.163, 0.403] | 0.116 [0.026, 0.158] | 0.129 [0.033, 0.192] | 0.000 [0.000, 0.000] |

Brightness profile realized by the surrogates (occupancy of alive-unit gates per K-decile, phase 2 pooled; decile 0 = smallest K):

| condition | source | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d9/d0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | actual | 0.0157 | 0.0104 | 0.0075 | 0.0064 | 0.0063 | 0.0052 | 0.0032 | 0.0029 | 0.0021 | 0.0008 | 0.052 |
| a003s0 | iso | 0.0251 | 0.0140 | 0.0108 | 0.0079 | 0.0072 | 0.0052 | 0.0037 | 0.0029 | 0.0019 | 0.0008 | 0.032 |
| a003s0 | isoc | 0.0203 | 0.0126 | 0.0091 | 0.0075 | 0.0067 | 0.0050 | 0.0035 | 0.0031 | 0.0019 | 0.0008 | 0.039 |
| a003s1 | actual | 0.0136 | 0.0099 | 0.0080 | 0.0068 | 0.0052 | 0.0043 | 0.0030 | 0.0026 | 0.0016 | 0.0006 | 0.042 |
| a003s1 | iso | 0.0236 | 0.0147 | 0.0104 | 0.0078 | 0.0064 | 0.0048 | 0.0037 | 0.0026 | 0.0015 | 0.0006 | 0.025 |
| a003s1 | isoc | 0.0201 | 0.0124 | 0.0091 | 0.0070 | 0.0055 | 0.0043 | 0.0036 | 0.0026 | 0.0018 | 0.0006 | 0.031 |
| a01s0 | actual | 0.0307 | 0.0229 | 0.0180 | 0.0169 | 0.0143 | 0.0118 | 0.0089 | 0.0080 | 0.0062 | 0.0035 | 0.115 |
| a01s0 | iso | 0.0479 | 0.0307 | 0.0249 | 0.0193 | 0.0178 | 0.0138 | 0.0105 | 0.0086 | 0.0063 | 0.0031 | 0.064 |
| a01s0 | isoc | 0.0417 | 0.0291 | 0.0206 | 0.0182 | 0.0148 | 0.0122 | 0.0097 | 0.0081 | 0.0054 | 0.0028 | 0.067 |
| a01s1 | actual | 0.0257 | 0.0210 | 0.0188 | 0.0172 | 0.0146 | 0.0128 | 0.0105 | 0.0091 | 0.0066 | 0.0032 | 0.125 |
| a01s1 | iso | 0.0466 | 0.0323 | 0.0248 | 0.0201 | 0.0173 | 0.0140 | 0.0115 | 0.0088 | 0.0059 | 0.0029 | 0.063 |
| a01s1 | isoc | 0.0384 | 0.0260 | 0.0212 | 0.0177 | 0.0153 | 0.0127 | 0.0110 | 0.0087 | 0.0064 | 0.0031 | 0.080 |
| a02s0 | actual | 0.0568 | 0.0460 | 0.0393 | 0.0378 | 0.0344 | 0.0296 | 0.0254 | 0.0228 | 0.0187 | 0.0112 | 0.197 |
| a02s0 | iso | 0.0830 | 0.0595 | 0.0505 | 0.0417 | 0.0390 | 0.0319 | 0.0261 | 0.0224 | 0.0175 | 0.0100 | 0.121 |
| a02s0 | isoc | 0.0739 | 0.0554 | 0.0455 | 0.0412 | 0.0351 | 0.0305 | 0.0265 | 0.0223 | 0.0170 | 0.0095 | 0.129 |
| a02s1 | actual | 0.0547 | 0.0451 | 0.0392 | 0.0378 | 0.0327 | 0.0297 | 0.0243 | 0.0219 | 0.0179 | 0.0103 | 0.188 |
| a02s1 | iso | 0.0784 | 0.0589 | 0.0480 | 0.0408 | 0.0363 | 0.0309 | 0.0266 | 0.0214 | 0.0157 | 0.0092 | 0.117 |
| a02s1 | isoc | 0.0713 | 0.0522 | 0.0440 | 0.0386 | 0.0339 | 0.0299 | 0.0256 | 0.0216 | 0.0165 | 0.0092 | 0.130 |
| a03s0 | actual | 0.0975 | 0.0809 | 0.0744 | 0.0700 | 0.0653 | 0.0600 | 0.0520 | 0.0488 | 0.0411 | 0.0293 | 0.300 |
| a03s0 | iso | 0.1294 | 0.1013 | 0.0896 | 0.0779 | 0.0739 | 0.0637 | 0.0547 | 0.0489 | 0.0407 | 0.0266 | 0.206 |
| a03s0 | isoc | 0.1219 | 0.0961 | 0.0849 | 0.0771 | 0.0690 | 0.0628 | 0.0559 | 0.0498 | 0.0407 | 0.0265 | 0.217 |
| a03s1 | actual | 0.0903 | 0.0797 | 0.0738 | 0.0722 | 0.0654 | 0.0597 | 0.0534 | 0.0490 | 0.0403 | 0.0277 | 0.307 |
| a03s1 | iso | 0.1276 | 0.1034 | 0.0885 | 0.0784 | 0.0720 | 0.0639 | 0.0571 | 0.0485 | 0.0385 | 0.0254 | 0.199 |
| a03s1 | isoc | 0.1196 | 0.0955 | 0.0854 | 0.0764 | 0.0693 | 0.0621 | 0.0571 | 0.0499 | 0.0403 | 0.0267 | 0.224 |
| a05s0 | actual | 0.2157 | 0.1976 | 0.1912 | 0.1868 | 0.1824 | 0.1753 | 0.1651 | 0.1576 | 0.1453 | 0.1196 | 0.554 |
| a05s0 | iso | 0.2601 | 0.2332 | 0.2198 | 0.2061 | 0.2006 | 0.1872 | 0.1742 | 0.1656 | 0.1516 | 0.1232 | 0.474 |
| a05s0 | isoc | 0.2551 | 0.2303 | 0.2161 | 0.2055 | 0.1973 | 0.1888 | 0.1783 | 0.1684 | 0.1539 | 0.1226 | 0.481 |
| a05s1 | actual | 0.2083 | 0.2019 | 0.1944 | 0.1937 | 0.1842 | 0.1792 | 0.1666 | 0.1573 | 0.1444 | 0.1143 | 0.549 |
| a05s1 | iso | 0.2586 | 0.2344 | 0.2183 | 0.2063 | 0.1986 | 0.1880 | 0.1784 | 0.1653 | 0.1486 | 0.1217 | 0.471 |
| a05s1 | isoc | 0.2507 | 0.2282 | 0.2155 | 0.2077 | 0.1985 | 0.1904 | 0.1799 | 0.1671 | 0.1519 | 0.1212 | 0.483 |
| ELU_s0_E1 | actual | 0.0412 | 0.0275 | 0.0217 | 0.0172 | 0.0191 | 0.0116 | 0.0076 | 0.0067 | 0.0050 | 0.0018 | 0.043 |
| ELU_s0_E1 | iso | 0.0390 | 0.0232 | 0.0181 | 0.0133 | 0.0122 | 0.0089 | 0.0064 | 0.0049 | 0.0033 | 0.0013 | 0.034 |
| ELU_s0_E1 | isoc | 0.0417 | 0.0264 | 0.0211 | 0.0163 | 0.0175 | 0.0111 | 0.0074 | 0.0062 | 0.0043 | 0.0016 | 0.038 |
| ELU_s0_E2 | actual | 0.0412 | 0.0275 | 0.0217 | 0.0172 | 0.0191 | 0.0116 | 0.0076 | 0.0067 | 0.0050 | 0.0018 | 0.043 |
| ELU_s0_E2 | iso | 0.0390 | 0.0232 | 0.0181 | 0.0133 | 0.0122 | 0.0089 | 0.0064 | 0.0049 | 0.0033 | 0.0013 | 0.034 |
| ELU_s0_E2 | isoc | 0.0417 | 0.0264 | 0.0211 | 0.0163 | 0.0175 | 0.0111 | 0.0074 | 0.0062 | 0.0043 | 0.0016 | 0.038 |
| ELU_s1_E1 | actual | 0.0342 | 0.0277 | 0.0236 | 0.0197 | 0.0144 | 0.0111 | 0.0089 | 0.0079 | 0.0048 | 0.0016 | 0.046 |
| ELU_s1_E1 | iso | 0.0374 | 0.0244 | 0.0178 | 0.0138 | 0.0115 | 0.0088 | 0.0069 | 0.0049 | 0.0029 | 0.0012 | 0.031 |
| ELU_s1_E1 | isoc | 0.0351 | 0.0276 | 0.0218 | 0.0174 | 0.0138 | 0.0099 | 0.0082 | 0.0070 | 0.0044 | 0.0014 | 0.040 |
| ELU_s1_E2 | actual | 0.0342 | 0.0277 | 0.0236 | 0.0197 | 0.0144 | 0.0111 | 0.0089 | 0.0079 | 0.0048 | 0.0016 | 0.046 |
| ELU_s1_E2 | iso | 0.0374 | 0.0244 | 0.0178 | 0.0138 | 0.0115 | 0.0088 | 0.0069 | 0.0049 | 0.0029 | 0.0012 | 0.031 |
| ELU_s1_E2 | isoc | 0.0351 | 0.0276 | 0.0218 | 0.0174 | 0.0138 | 0.0099 | 0.0082 | 0.0070 | 0.0044 | 0.0014 | 0.040 |

### 8.2 B1 — N_i: actual vs strat / flat / iso / iso+common (median [5%, 95%] over 20 draws)

| condition | pooled Fano: actual | strat | flat | iso | isoc | median F_i: actual | iso | isoc |
|---|---|---|---|---|---|---|---|---|
| a003s0 | 1.587 | 1.030 | 0.996 | 1.334 [1.268, 1.408] | 1.326 [1.239, 1.375] | 1.350 | 1.174 [1.114, 1.221] | 1.162 [1.067, 1.214] |
| a003s1 | 1.644 | 1.040 | 0.997 | 1.353 [1.283, 1.471] | 1.428 [1.310, 1.519] | 1.274 | 1.118 [1.000, 1.185] | 1.089 [1.000, 1.200] |
| a01s0 | 1.631 | 1.166 | 1.067 | 1.589 [1.496, 1.667] | 1.510 [1.444, 1.565] | 1.385 | 1.410 [1.351, 1.471] | 1.333 [1.274, 1.408] |
| a01s1 | 1.559 | 1.115 | 1.075 | 1.529 [1.452, 1.650] | 1.514 [1.434, 1.606] | 1.368 | 1.351 [1.278, 1.472] | 1.299 [1.277, 1.411] |
| a02s0 | 2.034 | 1.382 | 1.250 | 1.862 [1.812, 1.981] | 1.853 [1.796, 1.950] | 1.647 | 1.629 [1.572, 1.688] | 1.645 [1.540, 1.698] |
| a02s1 | 1.711 | 1.427 | 1.266 | 1.865 [1.790, 1.960] | 1.792 [1.745, 1.866] | 1.437 | 1.601 [1.526, 1.697] | 1.532 [1.483, 1.656] |
| a03s0 | 1.966 | 1.641 | 1.468 | 1.957 [1.855, 2.027] | 1.936 [1.844, 2.042] | 1.643 | 1.605 [1.527, 1.668] | 1.582 [1.496, 1.627] |
| a03s1 | 1.754 | 1.732 | 1.486 | 1.795 [1.735, 1.844] | 1.865 [1.825, 1.939] | 1.517 | 1.553 [1.463, 1.627] | 1.521 [1.450, 1.595] |
| a05s0 | 1.390 | 1.004 | 0.984 | 1.426 [1.391, 1.476] | 1.490 [1.458, 1.538] | 1.329 | 1.363 [1.324, 1.411] | 1.415 [1.376, 1.482] |
| a05s1 | 1.440 | 1.023 | 1.022 | 1.462 [1.428, 1.502] | 1.504 [1.471, 1.553] | 1.356 | 1.383 [1.331, 1.412] | 1.429 [1.395, 1.475] |
| ELU_s0_E1 | 3.620 | 1.308 | 1.056 | 1.898 [1.841, 1.967] | 3.477 [3.363, 3.600] | 2.851 | 1.510 [1.473, 1.620] | 2.814 [2.721, 2.913] |
| ELU_s0_E2 | 2.218 | 1.058 | 0.999 | 1.525 [1.471, 1.677] | 2.198 [2.125, 2.326] | 1.774 | 1.173 [1.066, 1.279] | 1.787 [1.690, 1.904] |
| ELU_s1_E1 | 4.866 | 1.291 | 1.064 | 2.112 [1.994, 2.207] | 4.916 [4.722, 5.052] | 3.986 | 1.695 [1.617, 1.827] | 3.686 [3.482, 3.802] |
| ELU_s1_E2 | 2.863 | 1.068 | 0.999 | 1.708 [1.586, 1.800] | 2.986 [2.888, 3.150] | 2.242 | 1.279 [1.218, 1.358] | 2.258 [2.064, 2.393] |

| condition | pooled mean: actual | iso | isoc | ratio strat | flat | iso | isoc | median μ_i: actual | iso | isoc |
|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.151 | 0.124 [0.115, 0.132] | 0.120 [0.113, 0.125] | 0.070 | 0.027 | 0.822 | 0.795 | 0.124 | 0.117 [0.105, 0.126] | 0.113 [0.103, 0.123] |
| a003s1 | 0.127 | 0.105 [0.097, 0.114] | 0.096 [0.093, 0.100] | 0.065 | 0.024 | 0.827 | 0.752 | 0.110 | 0.095 [0.087, 0.102] | 0.088 [0.076, 0.094] |
| a01s0 | 0.186 | 0.200 [0.194, 0.212] | 0.161 [0.154, 0.168] | 0.169 | 0.079 | 1.074 | 0.866 | 0.179 | 0.196 [0.183, 0.201] | 0.157 [0.147, 0.165] |
| a01s1 | 0.166 | 0.190 [0.180, 0.199] | 0.150 [0.143, 0.156] | 0.164 | 0.095 | 1.149 | 0.906 | 0.157 | 0.177 [0.172, 0.187] | 0.140 [0.129, 0.153] |
| a02s0 | 0.273 | 0.258 [0.242, 0.265] | 0.240 [0.231, 0.245] | 0.161 | 0.101 | 0.945 | 0.878 | 0.260 | 0.254 [0.236, 0.270] | 0.238 [0.222, 0.249] |
| a02s1 | 0.227 | 0.247 [0.242, 0.254] | 0.228 [0.214, 0.233] | 0.220 | 0.132 | 1.091 | 1.004 | 0.229 | 0.240 [0.232, 0.249] | 0.216 [0.206, 0.227] |
| a03s0 | 0.308 | 0.321 [0.309, 0.331] | 0.326 [0.310, 0.337] | 0.120 | 0.087 | 1.045 | 1.059 | 0.289 | 0.300 [0.288, 0.311] | 0.300 [0.278, 0.322] |
| a03s1 | 0.266 | 0.303 [0.293, 0.310] | 0.309 [0.301, 0.316] | 0.154 | 0.113 | 1.138 | 1.160 | 0.267 | 0.289 [0.278, 0.291] | 0.290 [0.278, 0.301] |
| a05s0 | 0.842 | 1.108 [1.086, 1.117] | 1.223 [1.190, 1.240] | 0.195 | 0.188 | 1.317 | 1.453 | 0.828 | 1.100 [1.077, 1.112] | 1.208 [1.183, 1.223] |
| a05s1 | 0.886 | 1.152 [1.136, 1.179] | 1.275 [1.254, 1.292] | 0.233 | 0.226 | 1.300 | 1.439 | 0.878 | 1.144 [1.111, 1.173] | 1.264 [1.244, 1.278] |
| ELU_s0_E1 | 0.549 | 0.132 [0.125, 0.137] | 0.554 [0.542, 0.578] | 0.039 | 0.012 | 0.241 | 1.010 | 0.511 | 0.120 [0.112, 0.130] | 0.514 [0.501, 0.552] |
| ELU_s0_E2 | 0.301 | 0.068 [0.065, 0.073] | 0.323 [0.305, 0.345] | 0.013 | 0.003 | 0.227 | 1.074 | 0.291 | 0.058 [0.057, 0.067] | 0.301 [0.282, 0.320] |
| ELU_s1_E1 | 0.736 | 0.160 [0.152, 0.169] | 0.711 [0.690, 0.751] | 0.028 | 0.009 | 0.217 | 0.967 | 0.675 | 0.143 [0.129, 0.157] | 0.668 [0.635, 0.701] |
| ELU_s1_E2 | 0.398 | 0.085 [0.080, 0.092] | 0.405 [0.395, 0.429] | 0.008 | 0.002 | 0.214 | 1.017 | 0.389 | 0.070 [0.067, 0.079] | 0.390 [0.364, 0.416] |

| condition | max N: actual | strat | flat | iso | isoc | #(N ≥ 10): actual | strat | flat | iso | isoc | targets μ=0 per draw: iso | isoc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 6 | 2 | 1 | 4 [4, 6] | 5 [4, 7] | 0 | 0 | 0 | 0 [0, 0] | 0 [0, 0] | 0.5 | 0.3 |
| a003s1 | 7 | 2 | 1 | 5 [4, 8] | 5 [4, 8] | 0 | 0 | 0 | 0 [0, 0] | 0 [0, 0] | 1.6 | 1.4 |
| a01s0 | 7 | 3 | 2 | 7 [6, 8] | 6 [5, 7] | 0 | 0 | 0 | 0 [0, 0] | 0 [0, 0] | 0.0 | 0.1 |
| a01s1 | 6 | 3 | 2 | 7 [6, 8] | 7 [5, 8] | 0 | 0 | 0 | 0 [0, 0] | 0 [0, 0] | 0.0 | 0.2 |
| a02s0 | 11 | 4 | 3 | 10 [8, 13] | 9 [8, 12] | 2 | 0 | 0 | 0 [0, 2] | 0 [0, 1] | 0.0 | 0.0 |
| a02s1 | 8 | 4 | 3 | 10 [8, 12] | 9 [7, 12] | 0 | 0 | 0 | 0 [0, 2] | 0 [0, 1] | 0.0 | 0.0 |
| a03s0 | 14 | 5 | 4 | 12 [9, 16] | 13 [10, 18] | 3 | 0 | 0 | 2 [0, 3] | 2 [1, 4] | 0.0 | 0.0 |
| a03s1 | 9 | 6 | 4 | 10 [8, 13] | 12 [10, 15] | 0 | 0 | 0 | 1 [0, 3] | 2 [1, 4] | 0.0 | 0.0 |
| a05s0 | 10 | 4 | 4 | 11 [10, 13] | 12 [11, 15] | 1 | 0 | 0 | 2 [1, 4] | 5 [3, 9] | 0.0 | 0.0 |
| a05s1 | 10 | 4 | 4 | 12 [10, 14] | 12 [10, 17] | 1 | 0 | 0 | 4 [2, 7] | 5 [2, 9] | 0.0 | 0.0 |
| ELU_s0_E1 | 18 | 3 | 2 | 8 [7, 11] | 18 [15, 21] | 32 | 0 | 0 | 0 [0, 1] | 33 [28, 41] | 0.2 | 0.0 |
| ELU_s0_E2 | 11 | 2 | 1 | 6 [5, 8] | 11 [9, 13] | 1 | 0 | 0 | 0 [0, 0] | 2 [0, 4] | 2.5 | 0.0 |
| ELU_s1_E1 | 28 | 3 | 2 | 9 [8, 11] | 28 [25, 31] | 80 | 0 | 0 | 0 [0, 1] | 73 [68, 85] | 0.3 | 0.0 |
| ELU_s1_E2 | 14 | 2 | 1 | 7 [6, 9] | 18 [14, 22] | 10 | 0 | 0 | 0 [0, 0] | 14 [8, 19] | 2.0 | 0.0 |

### 8.3 B1 — C_i: actual vs strat / flat / iso / iso+common (median [5%, 95%] over 20 draws)

| condition | pooled Fano: actual | strat | flat | iso | isoc | median F_i: actual | iso | isoc |
|---|---|---|---|---|---|---|---|---|
| a003s0 | 1.528 | 1.016 | 0.995 | 1.292 [1.250, 1.334] | 1.281 [1.246, 1.347] | 1.345 | 1.193 [1.153, 1.241] | 1.194 [1.154, 1.251] |
| a003s1 | 1.522 | 1.016 | 0.998 | 1.309 [1.260, 1.368] | 1.325 [1.269, 1.419] | 1.326 | 1.195 [1.144, 1.231] | 1.184 [1.149, 1.234] |
| a01s0 | 1.472 | 1.070 | 1.002 | 1.496 [1.459, 1.567] | 1.407 [1.383, 1.442] | 1.354 | 1.432 [1.346, 1.476] | 1.342 [1.308, 1.386] |
| a01s1 | 1.487 | 1.041 | 1.003 | 1.470 [1.415, 1.545] | 1.410 [1.360, 1.464] | 1.348 | 1.366 [1.304, 1.491] | 1.302 [1.235, 1.346] |
| a02s0 | 1.791 | 1.124 | 0.995 | 1.702 [1.655, 1.747] | 1.581 [1.533, 1.639] | 1.677 | 1.616 [1.574, 1.693] | 1.521 [1.460, 1.570] |
| a02s1 | 1.521 | 1.133 | 1.000 | 1.613 [1.598, 1.702] | 1.507 [1.475, 1.542] | 1.446 | 1.520 [1.471, 1.572] | 1.444 [1.394, 1.499] |
| a03s0 | 1.936 | 1.144 | 0.985 | 1.770 [1.706, 1.807] | 1.701 [1.673, 1.734] | 1.803 | 1.674 [1.612, 1.753] | 1.631 [1.596, 1.685] |
| a03s1 | 1.637 | 1.136 | 0.988 | 1.655 [1.603, 1.685] | 1.621 [1.573, 1.672] | 1.573 | 1.601 [1.534, 1.651] | 1.570 [1.524, 1.647] |
| a05s0 | 1.783 | 1.092 | 0.999 | 1.605 [1.556, 1.634] | 1.844 [1.780, 1.889] | 1.743 | 1.575 [1.518, 1.610] | 1.802 [1.722, 1.846] |
| a05s1 | 1.644 | 1.096 | 0.977 | 1.614 [1.560, 1.645] | 1.790 [1.736, 1.851] | 1.612 | 1.580 [1.511, 1.643] | 1.764 [1.685, 1.821] |
| ELU_s0_E1 | 3.904 | 1.200 | 1.075 | 1.695 [1.667, 1.752] | 3.868 [3.764, 3.972] | 3.489 | 1.564 [1.507, 1.599] | 3.460 [3.301, 3.572] |
| ELU_s0_E2 | 3.623 | 1.177 | 1.088 | 1.610 [1.554, 1.648] | 3.555 [3.455, 3.696] | 3.211 | 1.509 [1.438, 1.562] | 3.174 [3.004, 3.296] |
| ELU_s1_E1 | 5.071 | 1.189 | 1.068 | 1.841 [1.769, 1.895] | 5.288 [5.133, 5.488] | 4.501 | 1.637 [1.590, 1.722] | 4.547 [4.314, 4.775] |
| ELU_s1_E2 | 4.194 | 1.165 | 1.081 | 1.699 [1.657, 1.762] | 4.292 [4.187, 4.427] | 3.764 | 1.567 [1.508, 1.604] | 3.689 [3.543, 3.879] |

| condition | pooled mean: actual | iso | isoc | ratio strat | flat | iso | isoc | median μ_i: actual | iso | isoc |
|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.335 | 0.299 [0.285, 0.310] | 0.292 [0.283, 0.305] | 0.121 | 0.070 | 0.891 | 0.870 | 0.287 | 0.283 [0.271, 0.301] | 0.282 [0.270, 0.296] |
| a003s1 | 0.264 | 0.242 [0.231, 0.254] | 0.226 [0.219, 0.236] | 0.124 | 0.072 | 0.915 | 0.856 | 0.222 | 0.231 [0.217, 0.245] | 0.218 [0.205, 0.231] |
| a01s0 | 0.493 | 0.567 [0.554, 0.579] | 0.470 [0.457, 0.489] | 0.337 | 0.286 | 1.151 | 0.953 | 0.479 | 0.556 [0.532, 0.572] | 0.465 [0.446, 0.483] |
| a01s1 | 0.467 | 0.535 [0.524, 0.550] | 0.460 [0.449, 0.472] | 0.333 | 0.306 | 1.144 | 0.984 | 0.447 | 0.520 [0.509, 0.536] | 0.452 [0.437, 0.467] |
| a02s0 | 1.102 | 1.046 [1.025, 1.068] | 1.004 [0.975, 1.017] | 0.447 | 0.556 | 0.949 | 0.911 | 1.097 | 1.045 [1.028, 1.069] | 1.011 [0.983, 1.024] |
| a02s1 | 0.968 | 0.982 [0.967, 1.000] | 0.960 [0.931, 0.986] | 0.511 | 0.623 | 1.014 | 0.993 | 0.966 | 0.972 [0.943, 0.995] | 0.960 [0.933, 1.000] |
| a03s0 | 2.027 | 1.907 [1.877, 1.930] | 1.969 [1.937, 1.981] | 0.597 | 0.893 | 0.941 | 0.971 | 1.944 | 1.903 [1.854, 1.934] | 1.961 [1.915, 1.978] |
| a03s1 | 1.874 | 1.859 [1.829, 1.884] | 1.993 [1.969, 2.026] | 0.649 | 0.950 | 0.992 | 1.064 | 1.861 | 1.856 [1.816, 1.890] | 1.989 [1.961, 2.028] |
| a05s0 | 6.890 | 7.378 [7.294, 7.421] | 7.533 [7.465, 7.587] | 0.932 | 1.378 | 1.071 | 1.093 | 6.867 | 7.383 [7.249, 7.423] | 7.533 [7.474, 7.622] |
| a05s1 | 6.914 | 7.480 [7.390, 7.546] | 7.764 [7.701, 7.866] | 0.923 | 1.389 | 1.082 | 1.123 | 6.850 | 7.458 [7.349, 7.547] | 7.789 [7.664, 7.862] |
| ELU_s0_E1 | 1.968 | 0.522 [0.505, 0.534] | 1.842 [1.802, 1.878] | 0.084 | 0.065 | 0.265 | 0.936 | 1.953 | 0.511 [0.494, 0.535] | 1.831 [1.779, 1.867] |
| ELU_s0_E2 | 2.084 | 0.642 [0.629, 0.657] | 1.994 [1.951, 2.038] | 0.113 | 0.062 | 0.308 | 0.957 | 2.102 | 0.625 [0.608, 0.639] | 1.966 [1.929, 2.002] |
| ELU_s1_E1 | 2.215 | 0.535 [0.519, 0.550] | 2.002 [1.976, 2.057] | 0.067 | 0.054 | 0.241 | 0.904 | 2.221 | 0.514 [0.504, 0.532] | 1.988 [1.939, 2.055] |
| ELU_s1_E2 | 2.201 | 0.642 [0.620, 0.655] | 2.044 [1.999, 2.085] | 0.092 | 0.054 | 0.292 | 0.929 | 2.199 | 0.621 [0.589, 0.653] | 2.045 [1.999, 2.111] |

| condition | max C: actual | strat | flat | iso | isoc | #(C ≥ 10): actual | strat | flat | iso | isoc | targets μ=0 per draw: iso | isoc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 6 | 2 | 2 | 6 [5, 7] | 6 [5, 7] | 0 | 0 | 0 | 0 [0, 0] | 0 [0, 0] | 0.0 | 0.0 |
| a003s1 | 7 | 2 | 2 | 6 [5, 8] | 6 [5, 8] | 0 | 0 | 0 | 0 [0, 0] | 0 [0, 0] | 0.0 | 0.0 |
| a01s0 | 9 | 4 | 3 | 8 [7, 11] | 7 [6, 8] | 0 | 0 | 0 | 0 [0, 1] | 0 [0, 0] | 0.0 | 0.0 |
| a01s1 | 10 | 3 | 3 | 8 [7, 10] | 9 [7, 12] | 1 | 0 | 0 | 0 [0, 1] | 0 [0, 1] | 0.0 | 0.0 |
| a02s0 | 12 | 6 | 5 | 12 [11, 14] | 11 [9, 12] | 10 | 0 | 0 | 5 [2, 8] | 2 [0, 4] | 0.0 | 0.0 |
| a02s1 | 11 | 6 | 5 | 12 [10, 17] | 11 [10, 13] | 5 | 0 | 0 | 2 [1, 6] | 2 [1, 3] | 0.0 | 0.0 |
| a03s0 | 16 | 8 | 9 | 15 [13, 18] | 16 [14, 19] | 67 | 0 | 0 | 33 [21, 40] | 28 [22, 35] | 0.0 | 0.0 |
| a03s1 | 14 | 8 | 8 | 14 [13, 17] | 15 [13, 18] | 17 | 0 | 0 | 20 [15, 26] | 26 [20, 30] | 0.0 | 0.0 |
| a05s0 | 27 | 19 | 23 | 25 [23, 29] | 28 [25, 30] | 1818 | 1123 | 4287 | 2199 [2135, 2273] | 2375 [2300, 2464] | 0.0 | 0.0 |
| a05s1 | 27 | 19 | 23 | 25 [23, 28] | 27 [25, 30] | 1802 | 1088 | 4398 | 2312 [2232, 2397] | 2614 [2523, 2685] | 0.0 | 0.0 |
| ELU_s0_E1 | 22 | 5 | 4 | 11 [9, 12] | 24 [23, 28] | 259 | 0 | 0 | 1 [0, 2] | 208 [186, 226] | 0.0 | 0.0 |
| ELU_s0_E2 | 24 | 5 | 4 | 10 [9, 12] | 25 [23, 27] | 231 | 0 | 0 | 1 [0, 2] | 210 [191, 230] | 0.0 | 0.0 |
| ELU_s1_E1 | 33 | 5 | 3 | 11 [9, 13] | 32 [28, 37] | 393 | 0 | 0 | 1 [0, 4] | 341 [324, 360] | 0.0 | 0.0 |
| ELU_s1_E2 | 26 | 5 | 4 | 11 [9, 13] | 31 [28, 34] | 308 | 0 | 0 | 2 [0, 4] | 268 [261, 286] | 0.0 | 0.0 |

### 8.4 B2 — per-target position of the actual F_i and μ_i among the target's 20 draws

**N_i**

| condition | iso F: n | inside 90% | above q95 | below q05 | isoc F: n | inside 90% | above q95 | below q05 | iso μ_i: inside | above | below | isoc μ_i: inside | above | below |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 99 | 0.80 | 0.14 | 0.06 | 99 | 0.76 | 0.18 | 0.06 | 0.63 | 0.25 | 0.12 | 0.69 | 0.24 | 0.07 |
| a003s1 | 99 | 0.77 | 0.19 | 0.04 | 99 | 0.80 | 0.17 | 0.03 | 0.68 | 0.25 | 0.07 | 0.63 | 0.30 | 0.07 |
| a01s0 | 100 | 0.71 | 0.17 | 0.12 | 100 | 0.71 | 0.21 | 0.08 | 0.69 | 0.13 | 0.18 | 0.59 | 0.30 | 0.11 |
| a01s1 | 100 | 0.74 | 0.16 | 0.10 | 100 | 0.77 | 0.15 | 0.08 | 0.63 | 0.10 | 0.27 | 0.54 | 0.29 | 0.17 |
| a02s0 | 100 | 0.70 | 0.17 | 0.13 | 100 | 0.74 | 0.15 | 0.11 | 0.60 | 0.24 | 0.16 | 0.63 | 0.24 | 0.13 |
| a02s1 | 100 | 0.77 | 0.08 | 0.15 | 100 | 0.74 | 0.11 | 0.15 | 0.77 | 0.05 | 0.18 | 0.73 | 0.19 | 0.08 |
| a03s0 | 100 | 0.80 | 0.10 | 0.10 | 100 | 0.81 | 0.11 | 0.08 | 0.57 | 0.20 | 0.23 | 0.63 | 0.16 | 0.21 |
| a03s1 | 100 | 0.82 | 0.08 | 0.10 | 100 | 0.80 | 0.11 | 0.09 | 0.66 | 0.07 | 0.27 | 0.56 | 0.10 | 0.34 |
| a05s0 | 100 | 0.81 | 0.08 | 0.11 | 100 | 0.77 | 0.07 | 0.16 | 0.13 | 0.00 | 0.87 | 0.05 | 0.00 | 0.95 |
| a05s1 | 100 | 0.77 | 0.09 | 0.14 | 100 | 0.78 | 0.04 | 0.18 | 0.18 | 0.00 | 0.82 | 0.01 | 0.00 | 0.99 |
| ELU_s0_E1 | 100 | 0.36 | 0.63 | 0.01 | 100 | 0.84 | 0.09 | 0.07 | 0.03 | 0.97 | 0.00 | 0.81 | 0.10 | 0.09 |
| ELU_s0_E2 | 100 | 0.54 | 0.43 | 0.03 | 100 | 0.83 | 0.07 | 0.10 | 0.06 | 0.94 | 0.00 | 0.76 | 0.06 | 0.18 |
| ELU_s1_E1 | 100 | 0.20 | 0.80 | 0.00 | 100 | 0.83 | 0.10 | 0.07 | 0.02 | 0.98 | 0.00 | 0.67 | 0.21 | 0.12 |
| ELU_s1_E2 | 100 | 0.37 | 0.61 | 0.02 | 100 | 0.82 | 0.11 | 0.07 | 0.07 | 0.93 | 0.00 | 0.72 | 0.12 | 0.16 |

**C_i**

| condition | iso F: n | inside 90% | above q95 | below q05 | isoc F: n | inside 90% | above q95 | below q05 | iso μ_i: inside | above | below | isoc μ_i: inside | above | below |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 100 | 0.76 | 0.19 | 0.05 | 100 | 0.72 | 0.18 | 0.10 | 0.57 | 0.22 | 0.21 | 0.60 | 0.24 | 0.16 |
| a003s1 | 100 | 0.76 | 0.20 | 0.04 | 100 | 0.79 | 0.17 | 0.04 | 0.60 | 0.24 | 0.16 | 0.60 | 0.25 | 0.15 |
| a01s0 | 100 | 0.73 | 0.11 | 0.16 | 100 | 0.74 | 0.15 | 0.11 | 0.47 | 0.08 | 0.45 | 0.55 | 0.24 | 0.21 |
| a01s1 | 100 | 0.76 | 0.13 | 0.11 | 100 | 0.76 | 0.15 | 0.09 | 0.54 | 0.09 | 0.37 | 0.48 | 0.28 | 0.24 |
| a02s0 | 100 | 0.68 | 0.17 | 0.15 | 100 | 0.63 | 0.26 | 0.11 | 0.48 | 0.33 | 0.19 | 0.54 | 0.37 | 0.09 |
| a02s1 | 100 | 0.72 | 0.13 | 0.15 | 100 | 0.77 | 0.14 | 0.09 | 0.59 | 0.18 | 0.23 | 0.65 | 0.20 | 0.15 |
| a03s0 | 100 | 0.73 | 0.22 | 0.05 | 100 | 0.69 | 0.27 | 0.04 | 0.39 | 0.42 | 0.19 | 0.53 | 0.32 | 0.15 |
| a03s1 | 100 | 0.71 | 0.13 | 0.16 | 100 | 0.73 | 0.12 | 0.15 | 0.67 | 0.21 | 0.12 | 0.51 | 0.09 | 0.40 |
| a05s0 | 100 | 0.69 | 0.29 | 0.02 | 100 | 0.85 | 0.03 | 0.12 | 0.36 | 0.00 | 0.64 | 0.12 | 0.00 | 0.88 |
| a05s1 | 100 | 0.73 | 0.18 | 0.09 | 100 | 0.75 | 0.02 | 0.23 | 0.32 | 0.02 | 0.66 | 0.08 | 0.00 | 0.92 |
| ELU_s0_E1 | 100 | 0.08 | 0.92 | 0.00 | 100 | 0.93 | 0.03 | 0.04 | 0.00 | 1.00 | 0.00 | 0.64 | 0.33 | 0.03 |
| ELU_s0_E2 | 100 | 0.03 | 0.97 | 0.00 | 100 | 0.86 | 0.08 | 0.06 | 0.00 | 1.00 | 0.00 | 0.69 | 0.23 | 0.08 |
| ELU_s1_E1 | 100 | 0.06 | 0.94 | 0.00 | 100 | 0.85 | 0.06 | 0.09 | 0.00 | 1.00 | 0.00 | 0.40 | 0.56 | 0.04 |
| ELU_s1_E2 | 100 | 0.04 | 0.96 | 0.00 | 100 | 0.86 | 0.10 | 0.04 | 0.00 | 1.00 | 0.00 | 0.54 | 0.41 | 0.05 |

### 8.5 B3 — common-cause check against the iso / iso+common mean series (definitions as §5)

**N_i**

| condition | iso: Spearman (i,T) | within-target Spearman | var ratio [q25, q75] | noise-corr. | isoc: Spearman (i,T) | within-target Spearman | var ratio [q25, q75] | noise-corr. | strat Spearman (ref.) |
|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.353 | 0.345 | 0.2354 [0.1295, 0.3626] | 0.2008 | 0.366 | 0.370 | 0.2471 [0.1457, 0.3835] | 0.2146 | 0.240 |
| a003s1 | 0.324 | 0.319 | 0.1974 [0.1136, 0.4476] | 0.1649 | 0.332 | 0.334 | 0.2070 [0.1169, 0.3859] | 0.1679 | 0.204 |
| a01s0 | 0.416 | 0.404 | 0.4840 [0.2886, 0.7203] | 0.4507 | 0.395 | 0.409 | 0.3428 [0.1841, 0.5128] | 0.3084 | 0.359 |
| a01s1 | 0.390 | 0.379 | 0.4642 [0.2508, 0.7365] | 0.4327 | 0.366 | 0.374 | 0.3267 [0.1442, 0.5489] | 0.2957 | 0.340 |
| a02s0 | 0.489 | 0.483 | 0.4104 [0.2767, 0.6754] | 0.3796 | 0.493 | 0.501 | 0.3473 [0.2801, 0.5882] | 0.3180 | 0.440 |
| a02s1 | 0.458 | 0.456 | 0.5820 [0.3534, 0.8117] | 0.5504 | 0.464 | 0.459 | 0.5182 [0.3148, 0.7034] | 0.4762 | 0.443 |
| a03s0 | 0.484 | 0.478 | 0.4496 [0.2898, 0.7106] | 0.4213 | 0.482 | 0.475 | 0.4704 [0.2948, 0.6872] | 0.4332 | 0.399 |
| a03s1 | 0.469 | 0.467 | 0.4987 [0.3772, 0.6877] | 0.4521 | 0.460 | 0.455 | 0.4487 [0.3305, 0.7882] | 0.4095 | 0.405 |
| a05s0 | 0.509 | 0.510 | 0.5898 [0.4518, 0.6902] | 0.5449 | 0.518 | 0.504 | 0.7059 [0.5701, 0.8362] | 0.6507 | 0.340 |
| a05s1 | 0.510 | 0.516 | 0.5368 [0.4473, 0.6875] | 0.5019 | 0.525 | 0.520 | 0.6543 [0.5717, 0.7976] | 0.6106 | 0.364 |
| ELU_s0_E1 | 0.539 | 0.540 | 0.0538 [0.0303, 0.1087] | 0.0498 | 0.651 | 0.637 | 0.6995 [0.5540, 0.9374] | 0.6773 | 0.407 |
| ELU_s0_E2 | 0.396 | 0.395 | 0.0478 [0.0186, 0.1085] | 0.0403 | 0.550 | 0.536 | 0.5759 [0.4556, 0.7885] | 0.5486 | 0.206 |
| ELU_s1_E1 | 0.567 | 0.572 | 0.0499 [0.0310, 0.0925] | 0.0470 | 0.692 | 0.678 | 0.7561 [0.5684, 0.9330] | 0.7376 | 0.392 |
| ELU_s1_E2 | 0.431 | 0.454 | 0.0445 [0.0218, 0.0824] | 0.0398 | 0.596 | 0.584 | 0.6791 [0.4608, 0.9308] | 0.6595 | 0.165 |

**C_i**

| condition | iso: Spearman (i,T) | within-target Spearman | var ratio [q25, q75] | noise-corr. | isoc: Spearman (i,T) | within-target Spearman | var ratio [q25, q75] | noise-corr. | strat Spearman (ref.) |
|---|---|---|---|---|---|---|---|---|---|
| a003s0 | 0.346 | 0.332 | 0.2625 [0.1602, 0.3804] | 0.2224 | 0.378 | 0.347 | 0.2459 [0.1768, 0.3800] | 0.2098 | 0.129 |
| a003s1 | 0.326 | 0.293 | 0.2524 [0.1368, 0.3843] | 0.2170 | 0.331 | 0.333 | 0.2770 [0.1551, 0.4116] | 0.2331 | 0.113 |
| a01s0 | 0.419 | 0.420 | 0.4738 [0.3479, 0.6420] | 0.4310 | 0.397 | 0.401 | 0.3666 [0.2437, 0.5161] | 0.3311 | 0.277 |
| a01s1 | 0.375 | 0.382 | 0.4539 [0.3104, 0.6467] | 0.4082 | 0.347 | 0.332 | 0.3186 [0.2218, 0.4535] | 0.2815 | 0.226 |
| a02s0 | 0.523 | 0.540 | 0.4581 [0.3643, 0.5943] | 0.4274 | 0.536 | 0.554 | 0.3650 [0.2780, 0.4672] | 0.3395 | 0.372 |
| a02s1 | 0.453 | 0.457 | 0.4596 [0.3748, 0.6151] | 0.4282 | 0.484 | 0.478 | 0.4017 [0.2818, 0.5119] | 0.3674 | 0.347 |
| a03s0 | 0.560 | 0.556 | 0.4281 [0.3333, 0.5362] | 0.4081 | 0.593 | 0.596 | 0.4181 [0.3430, 0.5100] | 0.3948 | 0.415 |
| a03s1 | 0.557 | 0.556 | 0.4701 [0.3814, 0.6090] | 0.4428 | 0.566 | 0.571 | 0.4698 [0.4092, 0.5991] | 0.4376 | 0.406 |
| a05s0 | 0.639 | 0.642 | 0.5032 [0.4365, 0.5653] | 0.4819 | 0.714 | 0.715 | 0.6494 [0.5961, 0.7170] | 0.6242 | 0.420 |
| a05s1 | 0.606 | 0.600 | 0.5604 [0.4841, 0.6225] | 0.5309 | 0.699 | 0.694 | 0.7166 [0.6428, 0.7891] | 0.6851 | 0.426 |
| ELU_s0_E1 | 0.575 | 0.572 | 0.0510 [0.0315, 0.0857] | 0.0478 | 0.786 | 0.784 | 0.7017 [0.6351, 0.7935] | 0.6915 | 0.326 |
| ELU_s0_E2 | 0.539 | 0.541 | 0.0610 [0.0366, 0.0910] | 0.0564 | 0.792 | 0.789 | 0.6965 [0.6032, 0.8085] | 0.6855 | 0.246 |
| ELU_s1_E1 | 0.577 | 0.586 | 0.0417 [0.0280, 0.0563] | 0.0380 | 0.797 | 0.793 | 0.7594 [0.6039, 0.9052] | 0.7489 | 0.293 |
| ELU_s1_E2 | 0.530 | 0.535 | 0.0526 [0.0373, 0.0782] | 0.0480 | 0.791 | 0.790 | 0.7371 [0.6140, 0.8439] | 0.7241 | 0.165 |

### 8.6 Files

- `/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b/b_arrays_iso_<tag>.npz`: keys N, C (int16, −1 = not alive), Niso_mean, Ciso_mean (float, −1 = not alive), Niso_draws, Ciso_draws (int16 [r, T, i]), T_list (11..100).
- `.../agent_b/b_arrays_isoc_<tag>.npz`: same with prefix Nisoc_/Cisoc_.
- `results/xbar_arrangement_1014/b_iso_<tag>.json`, `b_isoc_<tag>.json` (per condition, incl. per-target arrays, widths, common-component sizes), `b_summary_iso.json` (this section's numbers). Logs `.../agent_b/logs/b_iso_<tag>.log`, `b_isoc_<tag>.log`.


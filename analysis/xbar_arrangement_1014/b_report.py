"""xbar_arrangement_1014 agent B: collect b_cond_<tag>.json + b_surr_<tag>.npz into b_summary.json and REPORT_B_stratified_null.md tables.
Extra per-surrogate numbers computed here from the stored count arrays: excess index (F_pool - 1)/mu_pool and the share of targets whose
phase-2 series is 0/1 only (max <= 1)."""
import os, json
import numpy as np, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
WT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014"
RES = os.path.join(WT, "results/xbar_arrangement_1014"); BIG = "/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b"
TAGS = ["a003s0", "a003s1", "a01s0", "a01s1", "a02s0", "a02s1", "a03s0", "a03s1", "a05s0", "a05s1", "ELU_s0_E1", "ELU_s0_E2", "ELU_s1_E1", "ELU_s1_E2"]
LEAKY = TAGS[:10]; ELU = TAGS[10:]


def extra(M, ids):
    """M (90, H) -> pooled excess index (F_pool - 1)/mu_pool, share of targets with max <= 1 and mu > 0, share with mu = 0."""
    pool = []; sparse = 0; zero = 0
    for i in ids:
        v = M[M[:, i] >= 0, i].astype(float); pool.append(v)
        if v.max() == 0: zero += 1
        elif v.max() <= 1: sparse += 1
    a = np.concatenate(pool); m = a.mean(); F = a.var(ddof=1) / m if m > 0 else np.nan
    return dict(excess=(F - 1) / m if m > 0 else np.nan, share_01=sparse / len(ids), share_zero=zero / len(ids))


def q(v):
    v = np.asarray(v, float); return dict(median=float(np.nanmedian(v)), q05=float(np.nanquantile(v, .05)), q95=float(np.nanquantile(v, .95)))


S = {}
for t in TAGS:
    d = json.load(open(os.path.join(RES, f"b_cond_{t}.json"))); z = np.load(os.path.join(BIG, f"b_surr_{t}.npz")); ids = list(z["ids"])
    d["extra"] = {}
    for qty in ("N", "C"):
        act = extra(z[f"{qty}_act"], ids); e = dict(actual=act)
        for null in ("strat", "flat"):
            arr = z[f"{qty}s_{null}"]; ex = [extra(arr[r], ids) for r in range(arr.shape[0])]
            e[null] = {k: q([x[k] for x in ex]) for k in ("excess", "share_01", "share_zero")}
        d["extra"][qty] = e
    S[t] = d


def inband(a, s): return s["q05"] <= a <= s["q95"]


def fm(x, n=3): return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def rng_(s, n=3, integer=False):
    if integer: return f"{s['median']:.0f} [{s['q05']:.0f}, {s['q95']:.0f}]"
    return f"{fm(s['median'], n)} [{fm(s['q05'], n)}, {fm(s['q95'], n)}]"


L = []
P = L.append
P("# REPORT B: brightness-stratified permutation null for N_i and C_i (xbar_arrangement_1014, agent B)\n")
P("2026-10-14. Box: 784-100-10, 1,200 fixed inputs, random labels, 100 tasks, saved task-end weights only (no training). All numbers [実測] unless marked. "
  "Scripts: `analysis/xbar_arrangement_1014/b_strat_null.py` (per condition), `b_report.py` (this file's tables), `b_jobs.txt` (the 14 conditions). "
  "JSON: `results/xbar_arrangement_1014/b_cond_<tag>.json` (per condition, incl. per-target arrays), `b_summary.json` (all tables). "
  "Count arrays (actual and 20+20 surrogates, phase 2): `/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b/b_surr_<tag>.npz`.\n")
P("## 1. Definitions (copied from `proj_004_drift/analysis/cobirth_seeds_1013/verify_counts.py`, function `counts`, verbatim)\n")
P("- ROOT = `obsidian-research-data/push_direction_proof_0930`; X = `load_X(d['subset'])` (1200×784, /255); Xt = [X, 1]; K_n = Xt_n·mean(Xt) (> 0, asserted).")
P("- State T = 1..100 (M1) = `end0[T−1]`, `end1[T−1]`; z_in = w_i·x_n + b_i; O_i = {n : z_in > 0}; k_i = |O_i|; alive = k_i > 0.")
P("- Threshold inputs T_i = top-3 of a score (stable argsort): leaky / E1: log K_n − |x| − 2 log(1 + e^{−|x|}) − log τ_i with x = z/τ_i, τ_i = 0.1 sd_n(z_in) + 1e−12; "
  "E2: log K_n + z_in on z_in < 0, −∞ on open inputs.")
P("- Bridge j → i (loose signature): j ≠ i, both alive, |O_j ∩ T_i| ≥ 2 and |O_j ∩ O_i| ≥ ⌈k_i/2⌉. N_i = number of bridges. C_i = #{j ≠ i alive : |O_j ∩ T_i| ≥ 2} (threshold pool; N_i ≤ C_i).")
P("- Phase 2 = T 11..100 (90 states). Targets = units with ≥ 30 alive phase-2 tasks (100 in all 14 conditions). μ_i = mean of N_i(T) over the target's alive phase-2 tasks; "
  "σ_i² ddof = 1; F_i = σ_i²/μ_i (undefined if μ_i = 0; dropped from medians, counts given). Pooled Fano = var(ddof=1)/mean over all targets' alive phase-2 tasks. "
  "max N and #(N ≥ 10) over the same pool. Same statistics for C_i. The cobirth_seeds `C_fano` is the population-variance pooled Fano over all alive phase-2 (i, T) (identical pool here).")
P("- Conditions (14): leaky `LR_ref_s{0,1}` (a = 0.1, `review/opus_bwdfloor/out/`), `LRa{0.03,0.2,0.3,0.5}_s{0,1}` (`review/opus_lrslope/out/`), gate leaky; "
  "ELU `ELU_ref_s{0,1}` (`review/opus_bwdfloor/out/`) with E1 and with E2.\n")
P("**Surrogates (new).** K-deciles = rank split of K over the 1,200 inputs into 10 blocks of 120. At each phase-2 state, for every unit the gate column o_j = 1[z_j· > 0] is replaced by "
  "a random permutation: **strat** = independently within each K-decile (preserves the unit's open count per decile); **flat** = over all 1,200 inputs (preserves k_j only). "
  "One permutation per (unit, state, surrogate), shared by all targets for which that unit is a candidate. The target's own T_i, O_i, k_i stay actual; the target is never its own candidate. "
  "R = 20 surrogates per null, generator `np.random.default_rng(1000 + r)` per (null, r), drawn state by state T = 11..100 (strat and flat use separate generators with the same seeds). "
  "Preservation asserted at r = 0, T = 11 (column sums; per-decile column sums for strat). Each surrogate gives a full phase-2 series N_i^(r)(T), C_i^(r)(T); the statistics above are computed per surrogate; "
  "tables give median [5%, 95%] over the 20 surrogates (np.quantile, linear).\n")
P("Criterion used in the verdicts: the null 'reproduces' a pooled statistic when the actual value lies inside the [5%, 95%] range of its 20 surrogate values.\n")

# ---- MATCH
P("## 2. Reproduction check against cobirth_seeds_1013 (verifier `vcounts_<tag>.npz`, `vstats_<tag>.json`; summary in `proj_004_drift/results/cobirth_seeds_1013/verify.json`)\n")
P("| condition | N cells equal (100×100) | C equal (alive cells) | k equal | F_i median own / ref | pooled Fano N own / ref | C_fano own / ref | max N own/ref | N≥10 own/ref | verdict |")
P("|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    m = S[t]["match"]; o = m["own"]; r = m["ref"]
    P(f"| {t} | {m['N_equal_all']} ({m['N_n_diff']} diff) | {m['C_equal_alive']} | {m['k_equal']} | {o['fano_median']:.4f} / {r['fano_median']:.4f} | {o['pool_fano']:.4f} / {r['pool_fano']:.4f} | "
      f"{o['C_fano_pop_all_alive']:.4f} / {r['C_fano']:.4f} | {o['max_N']}/{r['max_N']} | {o['n_N_ge10']}/{r['n_N_ge10']} | {m['verdict']} |")
fl = [S[t]["match"]["own"]["fano_median"] for t in LEAKY]; fe = [S[t]["match"]["own"]["fano_median"] for t in ELU]
cl = [S[t]["match"]["own"]["C_fano_pop_all_alive"] for t in LEAKY]; ce = [S[t]["match"]["own"]["C_fano_pop_all_alive"] for t in ELU]
nmatch = sum(S[t]["match"]["verdict"] == "MATCH" for t in TAGS)
P(f"\nRanges reproduced: N_i Fano median leaky {min(fl):.2f}–{max(fl):.2f} (ref 1.27–1.65), ELU {min(fe):.2f}–{max(fe):.2f} (ref 1.77–3.99); "
  f"C_i pooled Fano leaky {min(cl):.2f}–{max(cl):.2f} (ref 1.47–1.94), ELU {min(ce):.2f}–{max(ce):.2f} (ref 3.62–5.07). **{nmatch}/14 MATCH.**\n")

# ---- B1
for qty, name in (("N", "N_i (loose bridges)"), ("C", "C_i (threshold pool)")):
    P(f"## 3{'a' if qty == 'N' else 'b'}. B1 — {name}: actual vs strat vs flat (median [5%, 95%] over 20 surrogates)\n")
    P("| condition | F_i median: actual | strat | flat | pooled Fano: actual | strat | flat |")
    P("|---|---|---|---|---|---|---|")
    for t in TAGS:
        x = S[t][qty]; a = x["actual"]; s = x["strat"]["summary"]; f = x["flat"]["summary"]
        P(f"| {t} | {fm(a['F_median'])} | {rng_(s['F_median'])} | {rng_(f['F_median'])} | {fm(a['pool_fano'])} | {rng_(s['pool_fano'])} | {rng_(f['pool_fano'])} |")
    P("")
    P("| condition | μ_i median: actual | strat | flat | pooled mean: actual | strat | flat |")
    P("|---|---|---|---|---|---|---|")
    for t in TAGS:
        x = S[t][qty]; a = x["actual"]; s = x["strat"]["summary"]; f = x["flat"]["summary"]
        P(f"| {t} | {fm(a['mu_median'])} | {rng_(s['mu_median'])} | {rng_(f['mu_median'])} | {fm(a['mu_pooled'])} | {rng_(s['mu_pooled'])} | {rng_(f['mu_pooled'])} |")
    P("")
    P(f"| condition | max {qty}: actual | strat | flat | #({qty} ≥ 10): actual | strat | flat | excess (F_pool−1)/μ_pool: actual | strat | flat | targets μ=0 per surrogate: strat | flat | targets 0/1-only series: actual | strat | flat |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        x = S[t][qty]; a = x["actual"]; s = x["strat"]["summary"]; f = x["flat"]["summary"]; e = S[t]["extra"][qty]
        P(f"| {t} | {a['max']} | {rng_(s['max'], integer=True)} | {rng_(f['max'], integer=True)} | {a['n_ge10']} | {rng_(s['n_ge10'], integer=True)} | {rng_(f['n_ge10'], integer=True)} | "
          f"{fm(e['actual']['excess'])} | {rng_(e['strat']['excess'])} | {rng_(e['flat']['excess'])} | "
          f"{x['strat']['n_F_nan_total'] / 20:.1f} | {x['flat']['n_F_nan_total'] / 20:.1f} | {e['actual']['share_01']:.2f} | {rng_(e['strat']['share_01'], 2)} | {rng_(e['flat']['share_01'], 2)} |")
    P("")
P("Note [恒等式]: a 0/1 series of length n with m ones has F = n/(n−1)·(1 − m/n), i.e. ≈ 1 for sparse series; the last columns give the share of targets whose series is 0/1-only (μ > 0), "
  "so surrogate F_i medians near 1.000 are this arithmetic. The excess index (F_pool − 1)/μ_pool is listed because the surrogate means differ from the actual mean.\n")

# ---- B2
P("## 4. B2 — per-target position of the actual F_i (and μ_i) in the 20 surrogate values of the same target\n")
P("inside = actual within [5%, 95%] of the target's 20 surrogate values; above = actual > 95%; n = targets with finite actual F_i and finite surrogate quantiles.\n")
for qty in ("N", "C"):
    P(f"**{qty}_i**\n")
    P("| condition | strat F: n | inside 90% | above q95 | below q05 | flat F: n | inside 90% | above q95 | below q05 | strat μ_i: inside | above | flat μ_i: inside | above |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        x = S[t][qty]; bs = x["strat"]["B2_F"]; bf = x["flat"]["B2_F"]; ms = x["strat"]["B2_mu"]; mf = x["flat"]["B2_mu"]
        P(f"| {t} | {bs['n_used']} | {bs['frac_inside_90']:.2f} | {bs['frac_above_q95']:.2f} | {bs['frac_below_q05']:.2f} | {bf['n_used']} | {bf['frac_inside_90']:.2f} | {bf['frac_above_q95']:.2f} | {bf['frac_below_q05']:.2f} | "
          f"{ms['frac_inside_90']:.2f} | {ms['frac_above_q95']:.2f} | {mf['frac_inside_90']:.2f} | {mf['frac_above_q95']:.2f} |")
    P("")

# ---- B3
P("## 5. B3 — common-cause check: actual series vs the surrogate-mean series N̄_i(T) = mean over the 20 surrogates at the same T\n")
P("Spearman (average ranks) and Pearson over all (target, alive phase-2 T); within-target = median over targets of the correlation over T; "
  "var ratio = median over targets of Var_T(N̄)/Var_T(actual) (ddof 1) [q25, q75]; noise-corrected = (Var_T(N̄) − mean_T Var_r(N^(r))/20)/Var_T(actual), median.\n")
for qty in ("N", "C"):
    P(f"**{qty}_i**\n")
    P("| condition | strat: Spearman (i,T) | Pearson (i,T) | within-target Spearman | var ratio [q25, q75] | var ratio noise-corr. | flat: Spearman (i,T) | within-target Spearman | var ratio | noise-corr. |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        b = S[t][qty]["strat"]["B3"]; c = S[t][qty]["flat"]["B3"]
        P(f"| {t} | {b['spearman_iT']:.3f} | {b['pearson_iT']:.3f} | {b['within_target_spearman_median']:.3f} | {b['varratio_median']:.4f} [{b['varratio_q25']:.4f}, {b['varratio_q75']:.4f}] | {b['varratio_noisecorr_median']:.4f} | "
          f"{c['spearman_iT']:.3f} | {c['within_target_spearman_median']:.3f} | {c['varratio_median']:.4f} | {c['varratio_noisecorr_median']:.4f} |")
    P("")

# ---- B5
P("## 6. B5 — actual brightness profile preserved by strat (phase 2 pooled)\n")
P("occupancy(d) = open gates of alive units on decile-d inputs / (120 × alive units), pooled over T 11..100; decile 0 = smallest K. "
  "T_i share(d) = share of the alive targets' threshold inputs (3 per target-state) that lie in decile d.\n")
P("| condition | overall | " + " | ".join(f"occ d{d}" for d in range(10)) + " | d9/d0 |")
P("|---|---|" + "---|" * 11)
for t in TAGS:
    o = S[t]["B5"]["occupancy_by_decile"]
    P(f"| {t} | {S[t]['B5']['occupancy_overall']:.4f} | " + " | ".join(f"{v:.4f}" for v in o) + f" | {o[9] / o[0]:.3f} |")
P("")
P("| condition | " + " | ".join(f"T_i d{d}" for d in range(10)) + " |")
P("|---|" + "---|" * 10)
for t in TAGS:
    o = S[t]["B5"]["Ti_share_by_decile"]
    P(f"| {t} | " + " | ".join(f"{v:.3f}" for v in o) + " |")
P("")
P("K-decile mean K, decile 0 … decile 9 (the 1,200-input subset is identical within a seed in the files checked: LR_ref, ELU_ref, LRa0.03, LRa0.5): " + "; ".join(f"{t}: {S[t]['B5']['K_decile_mean'][0]:.1f}…{S[t]['B5']['K_decile_mean'][9]:.1f}" for t in ("a01s0", "a01s1")) + "\n")

# ---- verdicts (computed)
V = {}
for t in TAGS:
    v = {}
    for qty in ("N", "C"):
        a = S[t][qty]["actual"]
        for null in ("strat", "flat"):
            s = S[t][qty][null]["summary"]
            v[f"{qty}_{null}"] = dict(pool_fano=inband(a["pool_fano"], s["pool_fano"]), F_median=inband(a["F_median"], s["F_median"]),
                                      mu_median=inband(a["mu_median"], s["mu_median"]), mu_pooled=inband(a["mu_pooled"], s["mu_pooled"]),
                                      max=inband(a["max"], s["max"]), n_ge10=inband(a["n_ge10"], s["n_ge10"]))
    V[t] = v
json.dump(dict(conditions={t: {k: S[t][k] for k in ("tag", "npz", "gate", "match", "B5", "extra", "n_targets")} | {q_: {k: S[t][q_][k] for k in ("actual", "strat", "flat")} for q_ in ("N", "C")} for t in TAGS},
               inband=V, criterion="actual inside [q05, q95] of 20 surrogates"), open(os.path.join(RES, "b_summary.json"), "w"), indent=1)

# ---- verdict block (numbers computed here; inserted after the MATCH section)
def g(t, qty, who, key):
    return S[t][qty]["actual"][key] if who == "actual" else S[t][qty][who]["summary"][key]["median"]
def rr(vals): return f"{min(vals):.2f}–{max(vals):.2f}"
def inb(qty, null, key, tags): return sum(V[t][f"{qty}_{null}"][key] for t in tags)
VB = []; W = VB.append
W("## 2b. Verdicts (criterion: actual inside the [5%, 95%] range of the 20 surrogate values; ranges are over conditions, surrogate values are medians over 20)\n")
for qty, nm in (("C", "C_i"), ("N", "N_i")):
    for key, lab in (("pool_fano", "pooled Fano"), ("F_median", "median F_i")):
        W(f"- **strat reproduces actual {lab} of {nm}: {'yes' if inb(qty,'strat',key,TAGS)==14 else 'no'}** ({inb(qty,'strat',key,TAGS)}/14 in band). "
          f"actual leaky {rr([g(t,qty,'actual',key) for t in LEAKY])}, ELU {rr([g(t,qty,'actual',key) for t in ELU])}; "
          f"strat leaky {rr([g(t,qty,'strat',key) for t in LEAKY])}, ELU {rr([g(t,qty,'strat',key) for t in ELU])}; "
          f"flat leaky {rr([g(t,qty,'flat',key) for t in LEAKY])}, ELU {rr([g(t,qty,'flat',key) for t in ELU])} ({inb(qty,'flat',key,TAGS)}/14 in band).")
for qty, nm in (("N", "N_i"), ("C", "C_i")):
    ratio_s = [g(t, qty, "strat", "mu_pooled") / g(t, qty, "actual", "mu_pooled") for t in TAGS]
    ratio_f = [g(t, qty, "flat", "mu_pooled") / g(t, qty, "actual", "mu_pooled") for t in TAGS]
    ins = [S[t][qty]["strat"]["B2_mu"]["frac_inside_90"] for t in TAGS]
    W(f"- **strat reproduces μ_i of {nm}: {'yes' if inb(qty,'strat','mu_pooled',TAGS)==14 else 'no'}** (pooled mean in band {inb(qty,'strat','mu_pooled',TAGS)}/14; "
      f"strat/actual pooled mean leaky {rr(ratio_s[:10])}, ELU {rr(ratio_s[10:])}; per-target μ_i inside the strat band {rr(ins)} of targets). "
      f"flat/actual leaky {rr(ratio_f[:10])}, ELU {rr(ratio_f[10:])}.")
W(f"- **flat does not reproduce** pooled Fano (N {inb('N','flat','pool_fano',TAGS)}/14, C {inb('C','flat','pool_fano',TAGS)}/14), μ (N {inb('N','flat','mu_pooled',TAGS)}/14, C {inb('C','flat','mu_pooled',TAGS)}/14), "
  f"max N {inb('N','flat','max',TAGS)}/14, #N≥10 in band {inb('N','flat','n_ge10',TAGS)}/14 (all {inb('N','flat','n_ge10',TAGS)} are conditions with actual 0 and surrogate 0). "
  f"flat pooled Fano of C: {rr([g(t,'C','flat','pool_fano') for t in TAGS])} (cobirth null U: 0.98–1.09).")
spN = [S[t]['N']['strat']['B3']['spearman_iT'] for t in TAGS]; spC = [S[t]['C']['strat']['B3']['spearman_iT'] for t in TAGS]
vrN = [S[t]['N']['strat']['B3']['varratio_median'] for t in TAGS]; vrC = [S[t]['C']['strat']['B3']['varratio_median'] for t in TAGS]
W(f"- **common-cause correlation** (Spearman over (i, T), actual vs strat-mean): N leaky {rr(spN[:10])}, ELU {rr(spN[10:])}; C leaky {rr(spC[:10])}, ELU {rr(spC[10:])}. "
  f"Median Var_T(N̄^strat)/Var_T(N): leaky {min(vrN[:10]):.4f}–{max(vrN[:10]):.4f}, ELU {min(vrN[10:]):.4f}–{max(vrN[10:]):.4f}; for C: leaky {min(vrC[:10]):.4f}–{max(vrC[:10]):.4f}, ELU {min(vrC[10:]):.4f}–{max(vrC[10:]):.4f}. "
  f"flat Spearman: N {rr([S[t]['N']['flat']['B3']['spearman_iT'] for t in TAGS])}, C {rr([S[t]['C']['flat']['B3']['spearman_iT'] for t in TAGS])}.")
W(f"- **B4 (ELU)**: strat does not reproduce the larger ELU Fano. N pooled Fano actual {rr([g(t,'N','actual','pool_fano') for t in ELU])} vs strat {rr([g(t,'N','strat','pool_fano') for t in ELU])}; "
  f"C pooled Fano actual {rr([g(t,'C','actual','pool_fano') for t in ELU])} vs strat {rr([g(t,'C','strat','pool_fano') for t in ELU])} (leaky strat {rr([g(t,'C','strat','pool_fano') for t in LEAKY])}). "
  f"ELU max N actual {', '.join(str(g(t,'N','actual','max')) for t in ELU)} vs strat medians {', '.join(f"{g(t,'N','strat','max'):.0f}" for t in ELU)}; #N≥10 actual {', '.join(str(g(t,'N','actual','n_ge10')) for t in ELU)} vs strat 0 in all 80 surrogates.")
W("\n**Flags (numbers only).**\n")
W(f"1. The only in-band pooled statistic with nonzero actual value: N pooled Fano at a03s1 (actual {g('a03s1','N','actual','pool_fano'):.3f}, strat {rng_(S['a03s1']['N']['strat']['summary']['pool_fano'])}); "
  f"at a03s0 strat {rng_(S['a03s0']['N']['strat']['summary']['pool_fano'])} vs actual {g('a03s0','N','actual','pool_fano'):.3f}. Its strat pooled mean is {g('a03s1','N','strat','mu_pooled'):.3f} vs actual {g('a03s1','N','actual','mu_pooled'):.3f}.")
W(f"2. strat N means are {rr([g(t,'N','strat','mu_pooled')/g(t,'N','actual','mu_pooled') for t in LEAKY])}× actual (leaky) and "
  f"{rr([g(t,'N','strat','mu_pooled')/g(t,'N','actual','mu_pooled') for t in ELU])}× (ELU); strat F_i medians ≈ 1.000 coincide with 0/1-only series (share of targets whose strat series is 0/1-only with μ_i > 0: "
  f"{rr([S[t]['extra']['N']['strat']['share_01']['median'] for t in TAGS])}; targets with strat μ_i = 0 are dropped from the F_i median) [恒等式, see note in §3].")
W(f"3. flat C mean exceeds actual at a = 0.5 ({g('a05s0','C','flat','mu_pooled'):.2f}, {g('a05s1','C','flat','mu_pooled'):.2f} vs {g('a05s0','C','actual','mu_pooled'):.2f}, {g('a05s1','C','actual','mu_pooled'):.2f}) "
  f"and #(C ≥ 10) flat {g('a05s0','C','flat','n_ge10'):.0f}, {g('a05s1','C','flat','n_ge10'):.0f} vs actual {g('a05s0','C','actual','n_ge10')}, {g('a05s1','C','actual','n_ge10')}; "
  f"T_i share in decile 9: a0.5 {S['a05s0']['B5']['Ti_share_by_decile'][9]:.3f}, {S['a05s1']['B5']['Ti_share_by_decile'][9]:.3f}; a0.03 {S['a003s0']['B5']['Ti_share_by_decile'][9]:.3f}, {S['a003s1']['B5']['Ti_share_by_decile'][9]:.3f}; "
  f"occupancy d9/d0 from {min(S[t]['B5']['occupancy_by_decile'][9]/S[t]['B5']['occupancy_by_decile'][0] for t in TAGS):.3f} (ELU/a0.03) to {max(S[t]['B5']['occupancy_by_decile'][9]/S[t]['B5']['occupancy_by_decile'][0] for t in TAGS):.3f} (a0.5).")
W(f"4. Closest strat C mean: a = 0.5 (strat/actual {g('a05s0','C','strat','mu_pooled')/g('a05s0','C','actual','mu_pooled'):.3f}, {g('a05s1','C','strat','mu_pooled')/g('a05s1','C','actual','mu_pooled'):.3f}; "
  f"per-target μ_i inside the strat band {S['a05s0']['C']['strat']['B2_mu']['frac_inside_90']:.2f}, {S['a05s1']['C']['strat']['B2_mu']['frac_inside_90']:.2f}); its C pooled Fano is still {g('a05s0','C','strat','pool_fano'):.3f}, {g('a05s1','C','strat','pool_fano'):.3f} vs actual {g('a05s0','C','actual','pool_fano'):.3f}, {g('a05s1','C','actual','pool_fano'):.3f}.")
exC = "; ".join(t + " " + rng_(S[t]["extra"]["C"]["strat"]["excess"]) + " vs " + f"{S[t]['extra']['C']['actual']['excess']:.3f}" for t in ELU)
W(f"5. Excess index (F_pool − 1)/μ_pool of C, strat vs actual: {exC}; "
  f"for N the strat excess index has very wide ranges when μ ≈ 0 (e.g. ELU_s1_E2 {rng_(S['ELU_s1_E2']['extra']['N']['strat']['excess'])}), so it is not used in the verdicts.")
W(f"6. ELU E2 flat: F_i undefined for {100-S['ELU_s0_E2']['N']['flat']['B2_F']['n_used']} and {100-S['ELU_s1_E2']['N']['flat']['B2_F']['n_used']} targets (surrogate μ_i = 0 in all 20 flat draws); B2 fractions use the remaining targets.\n")
L[L.index("## 3a. B1 — N_i (loose bridges): actual vs strat vs flat (median [5%, 95%] over 20 surrogates)\n"):0] = VB
L.append("## 7. Files\n")
L.append("- per-condition arrays for the coordinator: `/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b/b_arrays_<tag>.npz`, keys N, C (int16, −1 = not alive), Nstrat_mean, Cstrat_mean, "
         "Nstrat_draws (int16 [r, T, i]), Nflat_mean, Cflat_mean (float, −1 = not alive), T_list (11..100); rows T 11..100, columns all 100 units; written by `b_arrays.py`.")
L.append("- full surrogate arrays (N and C, strat and flat, 20 draws each) + K + decile labels: `.../agent_b/b_surr_<tag>.npz`. Logs: `.../agent_b/logs/b_<tag>.log`.")
L.append("- `<tag>` ∈ " + ", ".join(TAGS) + ".\n")
with open(os.path.join(WT, "analysis/xbar_arrangement_1014/REPORT_B_stratified_null.md"), "w") as f: f.write("\n".join(L) + "\n")

cnt = lambda qty, null, key, tags: sum(V[t][f"{qty}_{null}"][key] for t in tags)
for qty in ("N", "C"):
    for null in ("strat", "flat"):
        print(qty, null, {key: (cnt(qty, null, key, LEAKY), cnt(qty, null, key, ELU)) for key in ("pool_fano", "F_median", "mu_median", "mu_pooled", "max", "n_ge10")})

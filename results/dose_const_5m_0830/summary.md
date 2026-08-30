# dose_const_5m_0830 summary

## Verdict

- Main: **BINARY_SATURATED_USE_LEVELS**
- Registered count verdict: **ONSET_SPREADS_BY_5M**
- Band: **BAND_MOVES_DOWN_BY_5M** (g*(1M)=10.04, g*(5M)=9.33)
- Binary saturation: 1M=False, 5M=True
- Jump J: 1M=0.457412, 5M=0.0288594

## Dose response

| arm | n onset 1M | n onset 5M | median log10 U 1M | median log10 U 5M | Δ median [percentile 95% CI] |
|---|---:|---:|---:|---:|---:|
| dose_off | 9 | 10 | -0.674832 | -0.09297 | 0.484514 [0.324196, 0.83892] |
| dose933 | 3 | 10 | -1.60124 | -0.215445 | 1.40201 [0.759941, 2.65358] |
| dose1004 | 8 | 10 | -0.995192 | -0.269169 | 0.731843 [0.485613, 1.20679] |
| dose1075 | 10 | 10 | -0.674896 | -0.236835 | 0.442623 [0.160101, 0.693532] |
| dose1146 | 10 | 10 | -0.640716 | -0.312646 | 0.373797 [0.258197, 0.613313] |
| dose1216 | 10 | 10 | -0.649239 | -0.274662 | 0.401789 [0.233207, 0.524272] |

0/10 の片側95%上限は 0.2589。
n_onset が全腕同一端点に飽和した窓は、水準側を判定基底として扱う。

## REPORT_ONLY at 5M

| arm | strict_dead | alive | eff_rank | ||mu|| | cos(mu, mu_off) |
|---|---:|---:|---:|---:|---:|
| dose_off | 99 | 1 | 1 | 3.20156 | 1 |
| dose933 | 98 | 2 | 1.09457 | 2.333 | 0.949957 |
| dose1004 | 99 | 1 | 1 | 2.51 | 0.974811 |
| dose1075 | 99 | 1 | 1 | 2.687 | 0.988613 |
| dose1146 | 99 | 1 | 1 | 2.864 | 0.9959 |
| dose1216 | 99 | 1 | 1 | 3.041 | 0.99878 |

## Sanity

- S0' bit reproduction: **PASS**
- S-pair preflight: **PASS**
- S-pair final: **PASS**
- S-dose preflight: **PASS**
- S-dose full logs: **PASS**
- S-tautology mutants: **PASS**
- S6 floor calibration: **PASS**
- 1M state hashes: **PASS**
- Exact-support identities: **PASS**

本介入は k を使うオラクル制御であり、学習手法ではない。
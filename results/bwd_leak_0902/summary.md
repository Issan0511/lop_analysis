# bwd_leak_0902 summary (stage all)

## Verdict

- **V1 (担い手): INCONCLUSIVE_DIVERGENCE**  — BL=missing, FL=present
- **V2 (折衷): INCONCLUSIVE_DIVERGENCE**  — BLW=missing, RW=present (BL=missing, source: this_run)
- V2 rows whose condition also held: —
- Raw onset triples (5M, n_onset per dose): BL=[]; FL=[10, 10]; RW=[10, 10]; BLW=[]; R=[10, 10]; LR=[0, 0]
- Numeric divergence: BLW_1216, BLW_933, BL_1216, BL_933

### 引用上の注意（spec §8）

- 0/10 は「5M までに観測しなかった」（片側 95% 上限 p<=0.2589）。
- **対照 `R_*` / `LR_*` は別走 `gate_dose_0830` の committed 値であり、
  同一走の腕ではない。** ペアリングは init・教師・入力実現までで、
  軌道は step 1 以降で分岐する。
- 復活数は **within-task**（同一タスク内かつ flip 不変）の定義でのみ
  「ReLU 腕は 0」。境界越えは支持の引き直しによる見かけの復活を含む。
- P5 の等価限界 0.15 dex は channel_2x2_0901 D4 からの継承で、
  **この系（1 層・std・床 1e-16・log10(mean U)）で較正し直していない**。
- P7c の 0.15 は s の単位であって dex ではない。P5 の 0.15 とは別の数。
- P6 はラベルを付けない。閾値が登録されておらず、R 系のベースラインが
  天井付近なので構成上すでに天井汚染される。

## Endpoints (5M)

| arm | act | wd_b | dose | onset 1M | onset 5M | median log10 U 1M | median log10 U 5M | source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BL_933 | bwd_leaky | 0.0 | 9.33 | — | — | — | — | NUMERIC_DIVERGENCE |
| BL_1216 | bwd_leaky | 0.0 | 12.16 | — | — | — | — | NUMERIC_DIVERGENCE |
| FL_933 | fwd_leaky | 0.0 | 9.33 | 3/10 | 10/10 | -1.64531 | -0.307489 | this run |
| FL_1216 | fwd_leaky | 0.0 | 12.16 | 10/10 | 10/10 | -0.565239 | -0.297672 | this run |
| RW_933 | relu | 0.001 | 9.33 | 2/10 | 10/10 | -3.84191 | -0.449667 | this run |
| RW_1216 | relu | 0.001 | 12.16 | 10/10 | 10/10 | -0.679554 | -0.15549 | this run |
| BLW_933 | bwd_leaky | 0.001 | 9.33 | — | — | — | — | NUMERIC_DIVERGENCE |
| BLW_1216 | bwd_leaky | 0.001 | 12.16 | — | — | — | — | NUMERIC_DIVERGENCE |
| R_933 | relu | 0.0 | 9.33 | 3/10 | 10/10 | -1.60124 | -0.215445 | gate_dose_0830 (別走) |
| R_1216 | relu | 0.0 | 12.16 | 10/10 | 10/10 | -0.649239 | -0.274662 | gate_dose_0830 (別走) |
| LR_933 | leaky_relu | 0.0 | 9.33 | 0/10 | 0/10 | -2.2272 | -2.60579 | gate_dose_0830 (別走) |
| LR_1216 | leaky_relu | 0.0 | 12.16 | 0/10 | 0/10 | -2.20796 | -2.65235 | gate_dose_0830 (別走) |

## Paired level contrasts at 5M (§5.2)

| endpoint | contrast | cross-run | n | median delta log10 U | percentile 95% CI | label | CI<0 | sign test p |
|---|---|---:|---:|---:|---|---|---:|---:|
| P3prime | BL_933_minus_R_933 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | BL_1216_minus_R_1216 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | FL_933_minus_R_933 | 1 | 10 | -0.104962 | [-0.20807, 0.0443776] | — | 0 | 0.7539 |
| P3prime | FL_1216_minus_R_1216 | 1 | 10 | -0.0136409 | [-0.151, 0.0534274] | — | 0 | 1 |
| P3prime | BLW_933_minus_R_933 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | BLW_1216_minus_R_1216 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | RW_933_minus_R_933 | 1 | 10 | -0.196622 | [-0.284262, -0.140625] | — | 0 | 0.001953 |
| P3prime | RW_1216_minus_R_1216 | 1 | 10 | 0.0525464 | [-0.0318507, 0.240912] | — | 0 | 0.3438 |
| P3prime_delta | BLW_933_minus_BL_933 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime_delta | BLW_1216_minus_BL_1216 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | BL_933_minus_LR_933 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | BL_1216_minus_LR_1216 | — | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | FL_933_minus_LR_933 | 1 | 10 | 2.32516 | [1.74235, 3.17439] | SHORT_OF_LR | 0 | 0.001953 |
| P5 | FL_1216_minus_LR_1216 | 1 | 10 | 2.30821 | [1.80551, 2.83671] | SHORT_OF_LR | 0 | 0.001953 |

## P6 (REPORT_ONLY, no label)

- dose 933: NOT_RUN
- dose 1216: NOT_RUN

Between-arm baseline spread: 3.27668 dex (early window, new arms: 2.74405; 1M window incl. committed controls: 3.27668; flag threshold 3.0). **P6 must not be read alone.**

## Sanity

- S1_omp: **PASS**
- S_cross: **PASS**
- S_bwd: **PASS**
- S_wd: **PASS**
- S_log: **PASS**
- S_ref: **PASS**
- S_limit_bwd: **PASS**
- S_limit_fwd: **PASS**
- S_limit_wd: **PASS**
- S_log_b: **PASS**
- S_pair: **PASS**
- S_dose: **PASS**
- S_taut: **PASS**
- S6_floor_inherited: **PASS**
- S_CI_degeneracy: **PASS**

# phantom_wall_0902 summary

## 型の verdict（§5.1）

| 型 | 単独 | +w-WD | RWw | verdict | 条件を満たしていた行 | 代替勾配 |
|---|---|---|---|---|---|---|
| BLR | diverged | diverged | present | **PHANTOM_DIVERGES** | PHANTOM_DIVERGES | -a (0.1) |
| BLQ | diverged | diverged | present | **PHANTOM_DIVERGES** | PHANTOM_DIVERGES | a_Q*z (0.01) |
| BLP | diverged | diverged | present | **PHANTOM_DIVERGES** | PHANTOM_DIVERGES | +a, mu-projected (0.1) |

- 対照 `RWw` の状態: **present**
- 飽和ガード: 発動せず
- Numeric divergence: BLP_1216, BLPw_1216, BLQ_1216, BLQw_1216, BLR_1216, BLRw_1216

### 引用上の注意

- **用量 12.16 の 1 点**の主張。引用時に用量を添える。
- 対照は**別走の committed 値**（`R_1216`/`LR_1216` は gate_dose_0830、
  `RW_1216` は bwd_leak_0902）。ペアリングは init・教師・入力実現までで、
  軌道は step 1 以降で分岐する。
- **`BLR`（a=0.1）と `BLQ`（a_Q=0.01）は係数が違う。** |gamma| が一致するのは
  z=-10 で、それより深いと `BLQ` の方が強い。両者の対比は登録していない。
- P8 は 3 型で第 2 項が同一なので、P8 どうしを引かない（追補 4）。
- P5 の等価限界 0.15 dex は**この系で較正していない**継承値。
- **`BLP` の µ 不変量は生の zbar**。s は denom 経由で動く（追補 8）。
- §1 の恒等式は**本 spec 初出の未登録の代数**であって既存ノートの定理ではない
  （spec §10.1 追補 1）。その代数は `BLR` について §7.2 と逆を指している
  （追補 3）が、9/2 裁定により予測は据え置いて凍結した。

## Endpoints (5M)

| arm | act | wd_w | onset 1M | onset 5M | median log10 U 5M | source |
|---|---|---:|---:|---:|---:|---|
| BLR_1216 | bwd_reflect | 0.0 | — | — | — | NUMERIC_DIVERGENCE |
| BLQ_1216 | bwd_quad | 0.0 | — | — | — | NUMERIC_DIVERGENCE |
| BLP_1216 | bwd_leaky_proj | 0.0 | — | — | — | NUMERIC_DIVERGENCE |
| BLRw_1216 | bwd_reflect | 0.0001 | — | — | — | NUMERIC_DIVERGENCE |
| BLQw_1216 | bwd_quad | 0.0001 | — | — | — | NUMERIC_DIVERGENCE |
| BLPw_1216 | bwd_leaky_proj | 0.0001 | — | — | — | NUMERIC_DIVERGENCE |
| RWw_1216 | relu | 0.0001 | 10/10 | 10/10 | 0 | this run |
| R_1216 | relu | 0.0 | 10/10 | 10/10 | -0.274662 | COMMITTED results/gate_dose_0830 |
| LR_1216 | leaky_relu | 0.0 | 0/10 | 0/10 | -2.65235 | COMMITTED results/gate_dose_0830 |
| RW_1216 | relu | 0.0 | 10/10 | 10/10 | -0.15549 | COMMITTED results/bwd_leak_0902 |

## Paired level contrasts at 5M (§5.2)

| endpoint | contrast | n | median delta | percentile 95% CI | label | CI<0 | sign p |
|---|---|---:|---:|---|---|---:|---:|
| P3prime | BLR_1216_minus_R_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | BLQ_1216_minus_R_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | BLP_1216_minus_R_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | BLRw_1216_minus_R_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | BLQw_1216_minus_R_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | BLPw_1216_minus_R_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P3prime | RWw_1216_minus_R_1216 | 10 | 0.209251 | [0.0940398, 0.397253] | — | 0 | 0.001953 |
| P5 | BLR_1216_minus_LR_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | BLQ_1216_minus_LR_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | BLP_1216_minus_LR_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | BLRw_1216_minus_LR_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | BLQw_1216_minus_LR_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |
| P5 | BLPw_1216_minus_LR_1216 | — | — | NUMERIC_DIVERGENCE | — | — | — |

## P8（REPORT_ONLY・ラベルなし）

- BLR: NUMERIC_DIVERGENCE
- BLQ: NUMERIC_DIVERGENCE
- BLP: NUMERIC_DIVERGENCE

## BLP の µ 不変性（追補 8: 不変量は zbar）

- BLP_1216: NUMERIC_DIVERGENCE
- BLPw_1216: NUMERIC_DIVERGENCE

## S-refl（**REPORT_ONLY へ降格**・追補 13）

登録時は必須ゲートだったが、実装ではなく本走の主 endpoint と同じ
物理の主張を検査するゲートだったため 9/2 に REPORT_ONLY へ降格した。
- 登録された期待（`BLR` の $\bar z$ が `BL` と逆向き）を満たしたか: **False**
- 30k 短走の平均 $\bar z$ ドリフト（**登録外の診断・転記対象ではない**）: BL +0.00614 (n=46), BLR +0.00044 (n=46)
- 実装側のゲート（S-cross・S-limit ×4・S-bwd・S-proj・S-wd-w・
  S-pair・S-dose・S-taut・S-ref）はすべて PASS している。

腕間ベースライン広がり（1M 窓・対照込み）: 1.77109 dex (閾値 3.0) — flagged=False

## Sanity

- S1_omp: **PASS**
- S_cross: **PASS**
- S_limit_static: **PASS**
- S_bwd: **PASS**
- S_proj: **PASS**
- S_wd_w: **PASS**
- S_ref: **PASS**
- S_limit_bwd_reflect: **PASS**
- S_limit_bwd_quad: **PASS**
- S_limit_bwd_leaky_proj: **PASS**
- S_limit_wd_w: **PASS**
- S_refl: **PASS**
- S_pair: **PASS**
- S_dose: **PASS**
- S_taut: **PASS**
- S6_floor_inherited: **PASS**
- S_CI_degeneracy: **PASS**

# mlp2_phase0b_0829 summary

## G-pre / G0

| arm | regime | verdict | n_onset | early seed median | late seed median [min, max] | dU_log median | CI degenerate |
|---|---|---|---:|---:|---:|---:|---:|
| L1w100 | INTERPOLATING | LOP_PRESENT | 10 | 6.86169e-05 | 0.807296 [0.336145, 0.999372] | 4.11925 | 1 |
| L1wide | INTERPOLATING | LOP_PRESENT | 9 | 3.41525e-10 | 0.302505 [0.00723322, 0.775846] | 8.75952 | 1 |
| L2 | INTERPOLATING | LOP_PRESENT | 10 | 3.00063e-13 | 0.0818685 [0.0574533, 0.661529] | 10.8492 | 1 |

全 seed の late U_k は verdict.csv に保存。腕間は非ペアとして扱った。

## Arm effects

回帰は全タスク末尾点を用い、腕ごとに独立な seed-cluster bootstrap を行った。

| x | contrast | coefficient | studentized 95% CI | degenerate | decision |
|---|---|---:|---:|---:|---|
| eff_rank | L1wide-L1w100 | 0.641159 | [0.455431, 0.844154] | 0 | H0_REJECTED |
| eff_rank | L2-L1w100 | -0.653655 | [-1.13236, -0.152889] | 0 | H0_REJECTED |
| alive | L1wide-L1w100 | -0.307793 | [-0.554599, -0.0889095] | 0 | H0_REJECTED |
| alive | L2-L1w100 | -3.16516 | [-3.82264, -2.47252] | 0 | H0_REJECTED |

## Wall depth D = -median(M)

| arm | layer | median seed Spearman(task,D) | studentized 95% CI | degenerate | increase |
|---|---:|---:|---:|---:|---:|
| L1w100 | 1 | 0.393081 | [0.369912, 0.415515] | 1 | 0 |
| L1wide | 1 | 0.295069 | [0.220301, 0.795696] | 1 | 0 |
| L2 | 1 | 0.359292 | [0.317384, 0.405353] | 1 | 0 |
| L2 | 2 | 0.0437927 | [0.0426766, 0.0497449] | 1 | 0 |

## Sanity

- S0: **PASS**
- S1/S2: **PASS**
- S3: **PASS**
- S5: **PASS**
- S6: **PASS**

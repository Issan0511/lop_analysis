# mlp2_phase1_0829 summary

候補 A（層入力の走行平均センタリング）。co-primary は P1（未フィット率の水準）と
P2（読み出し直前層の eff_rank の水準）。**両方が同じ向きを指したときだけ「効いた」**。

## 主 endpoint（§5.2）

| arm | baseline | pairing | P1 Δlog10 U（判定区間） | P1 改善 | P2 Δeff_rank（判定区間） | P2 改善 | verdict |
|---|---|---|---:|---:|---:|---:|---|
| L2_A1 | L2_none | paired | -0.915382 [-1.18818, -0.37278] | yes (percentile) | -1.33582 [-2.54354, 0.0788459] | no (percentile) | **INCONCLUSIVE_SPLIT** (P1_only) |
| L2_A2 | L2_none | paired | — (numeric divergence) | no (not computed) | — (numeric divergence) | no (not computed) | **NUMERIC_DIVERGENCE** |
| L2_Aall | L2_none | paired | -1.19116 [-1.44635, -0.985555] | yes (percentile) | 14.9684 [11.6829, 16.3962] | yes (percentile) | **A_EFFECTIVE** |
| L2_Aall | L2_A1 | paired | -0.389643 [-0.665511, -0.187932] | yes (percentile) | 15.173 [13.3364, 17.5374] | yes (percentile) | **A_EFFECTIVE** |
| L1w100_A1 | L2_none | unpaired | -1.54243 [-1.77889, -1.34315] | yes (percentile) | 13.6119 [12.3238, 15.4998] | yes (percentile) | **A_EFFECTIVE** |

## 数値発散（§5.7 実行追補）

| arm | detected step | task | seeds | 扱い |
|---|---:|---:|---|---|
| L2_A2 | 141000 | 15 | 7 | **NUMERIC_DIVERGENCE**（停止・救済なし） |

発散腕を含む対比は endpoint、CI、床割合を計算していない。発散腕を含まない対比だけを元の規則で集計した。

paired 対比の判定基底は censored -> sign_test、CI 退化 -> percentile、それ以外 -> studentized。unpaired 対比が検閲された場合は未登録の検定を足さない。
表の区間は**その行の判定に使った基底**のもの。studentized・percentile の両方と
符号検定 p、全 seed の Δ と水準は verdict.csv に保存してある。

## LoP 防止と水準シフトの判別（§5.2b）

| arm | baseline | Δearly | Δlate−early | trend ΔSpearman | signature |
|---|---|---:|---:|---:|---|
| L2_A1 | L2_none | 9.58894 | -10.6451 | -0.318527 | **NO_LOP_PREVENTION_SIGNATURE** |
| L2_A2 | L2_none | — | — | — | **NUMERIC_DIVERGENCE** |
| L2_Aall | L2_none | 9.8665 | -11.0433 | -0.261174 | **NO_LOP_PREVENTION_SIGNATURE** |
| L2_Aall | L2_A1 | 0.274768 | -0.676961 | 0.0556457 | **NO_LOP_PREVENTION_SIGNATURE** |
| L1w100_A1 | L2_none | 9.79425 | -11.417 | -0.532362 | **NO_LOP_PREVENTION_SIGNATURE** |

`LOP_PREVENTION_SIGNATURE` のときだけ「LoP を防いだ」と記述できる。この分類は主 verdict を上書きしない。

## 床検閲と CI 退化（§5.3 / §5.5）

| arm | 床割合(末尾窓) | baseline 床割合 | CENSORED | P1 CI退化 | P2 CI退化 | 符号検定 p (P1) |
|---|---:|---:|---:|---:|---:|---:|
| L2_A1 | 0 | 0 | 0 | 1 | 1 | 0.001953 |
| L2_A2 | — | — | — | — | — | — |
| L2_Aall | 0 | 0 | 0 | 1 | 1 | 0.001953 |
| L2_Aall | 0 | 0 | 0 | 1 | 1 | 0.1094 |
| L1w100_A1 | 0 | 0 | 0 | 1 | 1 | nan |

## 水準（末尾窓・REPORT_ONLY）

| arm | 未フィット率 median [min, max] |
|---|---:|
| L2_none | 0.13258 [0.061784, 0.526459] |
| L2_A1 | 0.0219155 [0.00369524, 0.106337] |
| L2_Aall | 0.00920405 [0.0040414, 0.024623] |
| L1w100_A1 | 0.00379596 [0.00348085, 0.00469359] |

| arm | layer | centered | eff_rank | alive | strict_dead | dose |
|---|---:|---:|---:|---:|---:|---:|
| L2_none | 1 | 0 | 9.80848 | 21.14 | 78.86 | 12.1254 |
| L2_none | 2 | 0 | 5.96681 | 12.09 | 87.91 | 8.48087 |
| L2_A1 | 1 | 1 | 19.5141 | 65.06 | 34.94 | 0.307793 |
| L2_A1 | 2 | 0 | 4.54016 | 9.32 | 90.68 | 7.18266 |
| L2_Aall | 1 | 1 | 3.13702 | 98.3 | 1.7 | 0.307793 |
| L2_Aall | 2 | 1 | 20.0103 | 46.34 | 53.66 | 1.03562 |
| L1w100_A1 | 1 | 1 | 19.5787 | 52.63 | 47.37 | 0.307793 |

## 整列（§5.2c・REPORT_ONLY）

| arm | layer | wcos_mean | eff_rank_W | stable_rank_W | top1_frac | sign_match_mean | sign_clone_frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2_none | 1 | 0.275767 | 15.7356 | 4.68808 | 0.213464 | 0.602826 | 0.00151313 |
| L2_none | 2 | 0.274677 | 67.8429 | 3.0392 | 0.329057 | 0.574492 | 0 |
| L2_A1 | 1 | 0.183181 | 19.2904 | 9.22545 | 0.108448 | 0.500933 | 0 |
| L2_A1 | 2 | 0.230144 | 71.5762 | 4.1579 | 0.240578 | 0.575844 | 0 |
| L2_Aall | 1 | 0.187141 | 18.7862 | 5.15738 | 0.194109 | 0.499338 | 0 |
| L2_Aall | 2 | 0.155807 | 72.8883 | 5.86906 | 0.170928 | 0.534414 | 0 |
| L1w100_A1 | 1 | 0.18511 | 19.172 | 8.56066 | 0.116886 | 0.500427 | 0 |

## dose の減衰（§5.4・REPORT_ONLY）

| arm | layer | centered | early median | late median | Δ median [95% CI] |
|---|---:|---:|---:|---:|---:|
| L2_none | 1 | 0 | 11.3739 | 12.1254 | 0.44947 [-0.156487, 1.00876] |
| L2_none | 2 | 0 | 12.2961 | 8.48087 | -4.06601 [-3.70047e+15, 3.24141] |
| L2_A1 | 1 | 1 | 0.285722 | 0.307793 | 0.0231377 [0.0112664, 0.0413827] |
| L2_A1 | 2 | 0 | 8.10849 | 7.18266 | -0.905294 [-0.932003, -0.874184] |
| L2_Aall | 1 | 1 | 0.285722 | 0.307793 | 0.0231377 [0.0112664, 0.0413827] |
| L2_Aall | 2 | 1 | 0.927876 | 1.03562 | 0.0968869 [0.0706482, 0.123695] |
| L1w100_A1 | 1 | 1 | 0.285722 | 0.307793 | 0.0231377 [0.0112664, 0.0413827] |

## 壁深さ D = -median(M)（§5.6）

**A を入れた層の D は恒真（TAUTOLOGICAL）。verdict にも機構の主張にも使わない。**

| arm | layer | centered | median seed Spearman(task,D) | 95% CI | 扱い |
|---|---:|---:|---:|---:|---|
| L2_none | 1 | 0 | 0.359292 | [0.317384, 0.405353] | REPORT_ONLY |
| L2_none | 2 | 0 | 0.0437927 | [0.0426766, 0.0497449] | REPORT_ONLY |
| L2_A1 | 1 | 1 | -0.00890068 | [-0.057628, 0.298217] | TAUTOLOGICAL |
| L2_A1 | 2 | 0 | 0.404979 | [0.395985, 0.420563] | REPORT_ONLY |
| L2_Aall | 1 | 1 | 0.000636483 | [-0.00657349, 0.00498687] | TAUTOLOGICAL |
| L2_Aall | 2 | 1 | 0.0196171 | [-0.336785, 0.0358288] | TAUTOLOGICAL |
| L1w100_A1 | 1 | 1 | -0.00601769 | [-0.0420934, 0.0168089] | TAUTOLOGICAL |

## Sanity（§4）

- S0'（L2_none == phase0b L2）: **PASS**
- S-pair（L2_* の対応づけ）: **PASS**
- S-pair-final（5M 後の env 一致）: **PASS**
- S-taut（A を入れた層の µ 項）: **PASS**
- S-copy（厳密記録の fork 検査）: **PASS**
- S1/S2（完走腕の32パターン厳密恒等式）: **PASS**
- S3（OMP_NUM_THREADS=1）: **PASS**
- S5（退化ガード自己検査）: **PASS**
- S7（数値発散検出器）: **PASS**
- S6（床較正）: **PASS**
- NUMERIC_DIVERGENCE 腕: L2_A2
- calibrated floor: 1e-23

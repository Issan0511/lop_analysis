# bias_wd_std_seediso_0901 — seed隔離再走の結果

事前登録: [`specs/spec_bias_wd_std_seediso_0901.md`](../../specs/spec_bias_wd_std_seediso_0901.md)。lambda・窓・判定境界は前走据え置き。

主判定は **INCONCLUSIVE_PARTIAL**。paired complete seeds = [0, 1, 2, 3, 4, 5, 6, 8, 9] (n=9)。

## Verdict

| pred | scope | verdict | evidence | ci_basis |
| --- | --- | --- | --- | --- |
| P-main | B10-B02 degradation ratio S_main/S_none | INCONCLUSIVE_PARTIAL | paired seeds=[0, 1, 2, 3, 4, 5, 6, 8, 9]; n=9; none drift +9.320376; main drift +2.976666; ratio 0.335409 CI [0.249288, 0.425399]; drift diff -6.343709 CI [-7.699630, -4.786038]; small denominator 0/9 | paired percentile |
| exclusion | S_none | ARM_VALID | excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; n=10; status=COMPLETE |  |
| exclusion | S_main | ARM_VALID | excluded=[7]; included=[0, 1, 2, 3, 4, 5, 6, 8, 9]; n=9; status=COMPLETE_WITH_EXCLUSIONS |  |
| exclusion | S_sub | ARM_VALID | excluded=[2]; included=[0, 1, 3, 4, 5, 6, 7, 8, 9]; n=9; status=COMPLETE_WITH_EXCLUSIONS |  |
| P-dose | B10-B02 degradation ratio S_sub/S_none | REPORT_ONLY | paired seeds=[0, 1, 3, 4, 5, 6, 7, 8, 9]; none drift +9.103595; sub drift +5.616816; ratio 0.655670 CI [0.501236, 0.821786]; diff -3.486779 CI [-5.325589, -1.578831] | paired percentile |
| dead | B10 strict_dead_frac S_none | REPORT_ONLY | L1 0.780020; L2 0.890900 |  |
| ledger | B02->B10 S_none L1 | REPORT_ONLY | seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; M -0.655741->-0.920601, delta -0.264859 CI [-0.344456, -0.189689]; B -0.241193->-0.211662, delta +0.029532 CI [-0.039690, +0.094726] | paired percentile |
| ledger | B02->B10 S_none L2 | REPORT_ONLY | seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; M -0.801581->-0.883125, delta -0.081544 CI [-0.189954, +0.036691]; B -0.211491->-0.192821, delta +0.018669 CI [-0.089521, +0.090885] | paired percentile |
| dead | B10 strict_dead_frac S_main | REPORT_ONLY | L1 0.681933; L2 0.530444 |  |
| ledger | B02->B10 S_main L1 | REPORT_ONLY | seeds=[0, 1, 2, 3, 4, 5, 6, 8, 9]; M -0.918390->-1.197665, delta -0.279275 CI [-0.351127, -0.201451]; B -0.013444->+0.003963, delta +0.017407 CI [+0.011005, +0.024047] | paired percentile |
| ledger | B02->B10 S_main L2 | REPORT_ONLY | seeds=[0, 1, 2, 3, 4, 5, 6, 8, 9]; M -1.060884->-1.199756, delta -0.138872 CI [-0.220176, -0.063133]; B -0.010887->+0.008701, delta +0.019588 CI [+0.011690, +0.026265] | paired percentile |
| dead | B10 strict_dead_frac S_sub | REPORT_ONLY | L1 0.690733; L2 0.715311 |  |
| ledger | B02->B10 S_sub L1 | REPORT_ONLY | seeds=[0, 1, 3, 4, 5, 6, 7, 8, 9]; M -0.926289->-1.190015, delta -0.263726 CI [-0.333907, -0.190680]; B -0.000007->-0.000003, delta +0.000004 CI [-0.000031, +0.000037] | paired percentile |
| ledger | B02->B10 S_sub L2 | REPORT_ONLY | seeds=[0, 1, 3, 4, 5, 6, 7, 8, 9]; undefined alive-median endpoint for seeds=[1] (no alive units in at least one block/channel); paired delta and CI not computed | not computed: undefined alive median |
| static | B10 level S_main-S_none | REPORT_ONLY | paired seeds=[0, 1, 2, 3, 4, 5, 6, 8, 9]; -2.480836 dex CI [-3.317507, -1.660915] | paired percentile |
| static | B10 level S_sub-S_none | REPORT_ONLY | paired seeds=[0, 1, 3, 4, 5, 6, 7, 8, 9]; -1.797789 dex CI [-2.937239, -0.680853] | paired percentile |

## 除外

| arm | status | excluded | included | n_included |
| --- | --- | --- | --- | --- |
| S_none | COMPLETE | [] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 10 |
| S_main | COMPLETE_WITH_EXCLUSIONS | [7] | [0, 1, 2, 3, 4, 5, 6, 8, 9] | 9 |
| S_sub | COMPLETE_WITH_EXCLUSIONS | [2] | [0, 1, 3, 4, 5, 6, 7, 8, 9] | 9 |

## B10（task 451–500）

| arm | mean_log10_unfit | log10_mean_unfit | L1_strict_dead_frac | L2_strict_dead_frac | L1_M_median_alive | L1_B_median_alive | L2_M_median_alive | L2_B_median_alive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_none | -1.81784 | -0.84033 | 0.78002 | 0.8909 | -0.920601 | -0.211662 | -0.883125 | -0.192821 |
| S_main | -4.32969 | -1.88684 | 0.681933 | 0.530444 | -1.19767 | 0.00396281 | -1.19976 | 0.00870109 |
| S_sub | -3.60786 | -1.31724 | 0.690733 | 0.715311 | -1.19001 | -2.56089e-06 | -1.27827 | 0.000143131 |

## 規約

- 各腕の除外上限2/10。3本目で `ARM_INVALID_EXCLUSION_LIMIT`
- seedの部分軌道は使わず、その腕から全時点を除外
- 主比較は完走seed共通集合、最低8本。bootstrap B=20000、seed=20260904
- B02=task 51–100、B10=task 451–500、床=1e-23
- 隔離後も10行分の入力乱数を消費し、非停止seedの対応軌道を維持

## 引いてはいけない線

- 除外後の結果を10/10完走と同一視しない。除外seedとnを常に併記する
- `LOP_PERSISTS` でもcenteredでのb-WD効果を否定しない
- `LOP_REMOVED` でもmu駆動説の棄却まで飛ばさない

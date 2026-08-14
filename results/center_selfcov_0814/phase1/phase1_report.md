# center_selfcov_0814 Phase 1: レジーム探索 (spec §4)

グリッド: th [None, 100] × width [5, 20] × kappa [1, 16] × lr [0.003, 0.01], c=0.0, seeds [0, 1], 200000 step

## セル別の選定指標 (seed 平均、srank_ratio 昇順)

  th  width  kappa    lr  srank_ratio  srank0  srank_end  dead_end  eval_ratio  eval_end  cos_e1W_end  wnorm_ratio  finite  n  c1_finite  c2_pathB  c3_notA  c4_lop
 100      5     16 0.010       0.7115  2.5304     1.7716       0.0      0.0906    0.1542       0.0631       1.0722    True  2       True      True     True   False
same     20     16 0.010       0.7389  6.5542     4.8481       0.0      0.0177    0.0667       0.0876       0.7870    True  2       True      True     True   False
 100     20     16 0.010       0.7699  6.5517     5.0682       0.0      0.0394    0.1038       0.1611       0.7776    True  2       True      True     True   False
 100      5      1 0.010       0.8026  2.8221     2.2536       0.0      0.0696    0.1374          NaN       0.9082    True  2       True     False    False   False
same     20     16 0.003       0.8708  6.1645     5.3736       0.0      0.0522    0.0971       0.1173       0.7224    True  2       True     False    False   False
 100     20     16 0.003       0.9036  6.1613     5.5750       0.0      0.0434    0.1105       0.3118       0.7141    True  2       True     False    False   False
same      5      1 0.010       0.9040  2.8307     2.5338       0.0      0.0435    0.0771          NaN       1.0328    True  2       True     False    False   False
same     20      1 0.010       0.9263  5.4868     5.0714       0.0      0.0376    0.0541          NaN       0.7578    True  2       True     False    False   False
same      5     16 0.003       0.9435  2.7561     2.6139       0.0      0.0349    0.0783       0.0208       0.9706    True  2       True     False    False   False
same      5      1 0.003       0.9678  2.9214     2.7869       0.0      0.0368    0.0465          NaN       0.9482    True  2       True     False    False   False
 100      5      1 0.003       0.9781  2.9216     2.8035       0.0      0.0908    0.1354          NaN       0.7207    True  2       True     False    False   False
same     20      1 0.003       0.9888  6.1696     6.0895       0.0      0.0780    0.1219          NaN       0.7538    True  2       True     False    False   False
 100     20      1 0.003       1.0057  6.1653     6.1913       0.0      0.0772    0.1274          NaN       0.7381    True  2       True     False    False   False
same      5     16 0.010       1.0517  2.5381     2.6690       0.0      0.0567    0.0998       0.1361       1.0845    True  2       True     False    False   False
 100      5     16 0.003       1.0819  2.7596     2.8921       0.0      0.1134    0.1498       0.4541       0.7802    True  2       True     False    False   False
 100     20      1 0.010       1.0862  5.4842     5.9602       0.0      0.0380    0.0605          NaN       0.7398    True  2       True     False    False   False


- 基準1 (発散なし) 通過: 16/16
- 基準2 (srank ≤ 80%) 通過: 3/16
- 基準3 (dead < 0.5) 通過: 3/16
- 基準4 (eval_loss 2倍以上) 通過: 0/16

## 採用セル

 th width kappa    lr srank_ratio    srank0 srank_end dead_end eval_ratio  eval_end cos_e1W_end wnorm_ratio finite n c1_finite c2_pathB c3_notA c4_lop
100     5    16  0.01    0.711511  2.530425  1.771555      0.0   0.090621  0.154205    0.063098    1.072168   True 2      True     True    True  False

注: 基準4 (LoP 発現) を満たすセルが無いため、基準3 までを満たすセルのうち srank 低下最大を採用した (要 summary 明記)。

- status: OK_NO_LOP

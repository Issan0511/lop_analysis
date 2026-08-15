# condA_freeze_0815 — 条件A の freeze_bias 腕 (レベル1: 必要性)

判定基準 PA-1..PA-3 は configs/condA_freeze_0815.yaml のヘッダに実行前から事前登録済み。

## 判定

pred                                 scope       verdict                                                                                                                                                                                            evidence
PA-1 条件A で b 凍結が dead を減らす (frozen < free)          FAIL                                                                  w5: free 0.800 → frozen 0.720 (diff -0.080 CI [-0.240, +0.000]); w100: free 0.968 → frozen 0.970 (diff +0.002 CI [-0.004, +0.008])
PA-2           寄与の大きさ (減少率 < 0.2 なら b は脇役)         MINOR                                                                                                                        w5: 減少率 0.100; w100: 減少率 -0.002。→ **b は脇役** (Phase 0 の「b 主導 dead は 5.3%」と整合)
PA-3 b 凍結の機能的影響 (eval_loss, frozen − free) MIXED_OR_NULL w5: free 0.8485 → frozen 0.7499 (diff -0.0986 CI [-0.1943, -0.0101]); w100: free 0.9374 → frozen 1.0583 (diff +0.1209 CI [+0.0141, +0.2804])。留保: frozen は「死のない同一ネット」ではなく閾値表現力ごと奪ったネットなので厳密な反実仮想ではない


## 腕別の最終値

 width  dead_free  dead_frozen  reduction  dead_diff  dead_lo  dead_hi  eval_free  eval_frozen  eval_diff  eval_lo  eval_hi  b_mean_free  b_min_free  n
     5      0.800         0.72    0.10000     -0.080   -0.240    0.000    0.84851      0.74988   -0.09863 -0.19433 -0.01013      0.29438    -1.56225  5
   100      0.968         0.97   -0.00207      0.002   -0.004    0.008    0.93742      1.05829    0.12087  0.01409  0.28040     -0.34306    -1.68953  5


## 位置づけ

- b 凍結は条件A の dead を有意に変えない → **b は既存 LoP に不要**。bias_margin_0814 の機構は µ=0 レジーム限定と結論できる。
- とくに w100 は diff +0.002 CI [-0.004, +0.008] で**効果が完全にゼロ**。µ≠0 では b を 0 に固定しても dead は同じだけ進む。
- w5 は diff -0.080 CI [-0.240, +0.000] で減少方向だが CI がゼロを含む (境界)。w5 は dead_frac が 0.2 刻みに量子化される (5 ユニット) ためn=5 seed の bootstrap は粗く、**示唆どまりで有意ではない**。

### 主張への反映
- 「b が既存の LoP に関与する」は**言えない** (レベル1 不成立)。
- bias_margin_0814 で言えるのは**レベル0 限定**: 「µ 経路を塞いだ µ=0 設定では b が margin の唯一のノブになり dead を作れる」。既存 LoP (µ≠0) の説明にはならない。
- これは仮説の否定ではなく**適用範囲の確定**。Phase 0 の静的分解 (b 主導 dead 5.3%) と本テスト (凍結しても dead 不変) が同じ方向を指しており、条件A の dead は µ 経路で完結している。

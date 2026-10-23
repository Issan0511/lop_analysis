# SiLU L2の相2予測と実際の合計の照合

**課題100末の切替では、相2予測がL3を浮かせL2を沈める両方の向きと対応した。課題50末のL2や現課題fitには同じ対応がなく、長期の沈降死まで相2で説明できたとはしない。**

課題100末・保存Adam分母固定の全経路では、L2の符号一致94.7%、両方向的中率86.9%、Spearman .939。L3は99.1%、99.5%、.882。L2の自unit up行の更新だけでは平均は浮かせる側で、上流経路を足した合計は沈める側になる。この符号関係を予測・実測の双方が示した。単に最終FFNが誤差を増幅するという量の観察から推定したものではない。

対象は最終FFN L3が正側へ出て、その一つ前L2が沈んだseed1000。SiLU 7状態、SiLUcap 4状態。保存状態だけ、学習更新0。10seedの一般性は測っていない。

相2の全head共同fitから予測した勾配を、全パラメータ更新によるrの変化に運ぶ。実測も全勾配であり、他unit・注意・FFN・LN・上流のパラメータ更新を含む。正のSが沈める側（Δr=−ηS）。

## 登録判定

| 分母 | 切替への対応 | 現課題fitへの対応 | 後期の合計が沈める側 |
|---|---|---|---|
| SGD | PARTIAL (1/2) | NOT_MATCHED (0/2) | False |
| Adam_fixed | PARTIAL (1/2) | NOT_MATCHED (0/2) | False |

対応の条件は課題50末・100末の両方でbalanced accuracy≥.65。閾値不達はこの有限状態への適用の不一致であり、旧ラベル平均の相2理論そのものの反証ではない。

## SiLU L2・全経路の符号照合

| 保存点 | 分母 | 実勾配 | 共通非ゼロ数 | 相2の沈む割合 | 実測の沈む割合 | 符号一致 | 両方向的中率 | Spearman | 相2の全unit平均 | 実測の全unit平均 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1末 | Adam_fixed | fit_full | 512 | 23.0% | 44.1% | 49.6% | 46.4% | -0.103 | 浮かせる (-1.03e+04) | 浮かせる (-2.98) |
| 1末 | Adam_fixed | switch_full | 512 | 23.0% | 87.1% | 25.4% | 39.7% | -0.190 | 浮かせる (-1.03e+04) | 沈める (596) |
| 1末 | SGD | fit_full | 512 | 40.4% | 43.8% | 47.9% | 46.6% | -0.104 | 浮かせる (-1.24) | 浮かせる (-0.00101) |
| 1末 | SGD | switch_full | 512 | 40.4% | 85.7% | 43.0% | 49.6% | 0.003 | 浮かせる (-1.24) | 沈める (0.178) |
| 5末 | Adam_fixed | fit_full | 512 | 54.9% | 60.2% | 51.4% | 50.4% | -0.025 | 沈める (738) | 沈める (1.57) |
| 5末 | Adam_fixed | switch_full | 512 | 54.9% | 27.5% | 32.4% | 30.7% | -0.528 | 沈める (738) | 浮かせる (-110) |
| 5末 | SGD | fit_full | 512 | 67.6% | 50.0% | 41.4% | 41.4% | -0.247 | 沈める (1.71) | 沈める (0.000146) |
| 5末 | SGD | switch_full | 512 | 67.6% | 33.8% | 23.2% | 26.5% | -0.708 | 沈める (1.71) | 浮かせる (-0.0393) |
| 10末 | Adam_fixed | fit_full | 512 | 34.8% | 28.5% | 49.2% | 41.0% | -0.242 | 浮かせる (-1.82e+03) | 浮かせる (-3.36) |
| 10末 | Adam_fixed | switch_full | 512 | 34.8% | 25.4% | 71.5% | 68.5% | 0.565 | 浮かせる (-1.82e+03) | 浮かせる (-90.6) |
| 10末 | SGD | fit_full | 512 | 54.1% | 36.5% | 41.4% | 41.9% | -0.205 | 沈める (0.262) | 浮かせる (-0.00209) |
| 10末 | SGD | switch_full | 512 | 54.1% | 35.4% | 52.3% | 53.9% | 0.054 | 沈める (0.262) | 浮かせる (-0.0461) |
| 25末 | Adam_fixed | fit_full | 512 | 98.0% | 57.4% | 57.4% | 50.3% | -0.008 | 沈める (4.04e+03) | 沈める (0.562) |
| 25末 | Adam_fixed | switch_full | 512 | 98.0% | 10.2% | 11.7% | 50.0% | -0.136 | 沈める (4.04e+03) | 浮かせる (-46.6) |
| 25末 | SGD | fit_full | 512 | 96.9% | 59.2% | 56.8% | 48.2% | -0.424 | 沈める (3.08) | 沈める (0.00275) |
| 25末 | SGD | switch_full | 512 | 96.9% | 37.9% | 40.2% | 51.7% | 0.444 | 沈める (3.08) | 浮かせる (-0.0465) |
| 50末 | Adam_fixed | fit_full | 512 | 92.8% | 55.1% | 54.9% | 50.5% | 0.067 | 沈める (2.9e+03) | 沈める (0.256) |
| 50末 | Adam_fixed | switch_full | 512 | 92.8% | 28.1% | 33.0% | 52.1% | 0.260 | 沈める (2.9e+03) | 浮かせる (-14.2) |
| 50末 | SGD | fit_full | 512 | 84.2% | 32.6% | 32.8% | 44.0% | -0.354 | 沈める (1.91) | 浮かせる (-0.002) |
| 50末 | SGD | switch_full | 512 | 84.2% | 47.3% | 57.6% | 59.5% | 0.528 | 沈める (1.91) | 浮かせる (-0.00279) |
| 100更新100 | Adam_fixed | fit_full | 429 | 77.9% | 47.6% | 39.9% | 41.2% | -0.257 | 沈める (4.94e+03) | 浮かせる (-1.2) |
| 100更新100 | Adam_fixed | switch_full | 429 | 77.9% | 72.3% | 94.4% | 89.9% | 0.917 | 沈める (4.94e+03) | 沈める (129) |
| 100更新100 | SGD | fit_full | 429 | 86.2% | 57.6% | 57.3% | 51.9% | -0.010 | 沈める (5.99) | 沈める (0.0137) |
| 100更新100 | SGD | switch_full | 429 | 86.2% | 66.2% | 74.8% | 64.6% | 0.611 | 沈める (5.99) | 沈める (0.0991) |
| 100末 | Adam_fixed | fit_full | 489 | 85.1% | 31.3% | 26.6% | 38.0% | -0.341 | 沈める (6.24e+03) | 浮かせる (-1.84) |
| 100末 | Adam_fixed | switch_full | 489 | 85.1% | 79.8% | 94.7% | 86.9% | 0.939 | 沈める (6.24e+03) | 沈める (141) |
| 100末 | SGD | fit_full | 489 | 88.1% | 31.7% | 23.1% | 35.1% | -0.635 | 沈める (6.69) | 浮かせる (-0.0146) |
| 100末 | SGD | switch_full | 489 | 88.1% | 75.3% | 85.9% | 72.3% | 0.753 | 沈める (6.69) | 沈める (0.188) |

沈む割合は応答ありかつ両者非ゼロの共通集合。末尾2列は全512unitの平均。BAは実測で沈む群・浮く群の的中率の平均で、各10unit未満なら未定義。係数の大きさは模型間で比較せず、平均の符号を見る。

## 全経路・自分のup行・上流の違い（SiLU、課題100末）

| 層 | 分母 | 経路 | 実勾配 | 符号一致 | 両方向的中率 | ρ | 相2平均 | 実測平均 |
|---|---|---|---|---|---|---|---|---|
| L2 | Adam_fixed | full | fit_full | 26.6% | 38.0% | -0.341 | 沈める | 浮かせる |
| L2 | Adam_fixed | full | switch_full | 94.7% | 86.9% | 0.939 | 沈める | 沈める |
| L2 | Adam_fixed | own | fit_full | 47.6% | 48.1% | -0.011 | 浮かせる | 沈める |
| L2 | Adam_fixed | own | switch_full | 71.4% | 70.7% | 0.622 | 浮かせる | 浮かせる |
| L2 | Adam_fixed | upstream | fit_full | 27.0% | 39.7% | -0.372 | 沈める | 浮かせる |
| L2 | Adam_fixed | upstream | switch_full | 94.5% | 86.3% | 0.941 | 沈める | 沈める |
| L2 | SGD | full | fit_full | 23.1% | 35.1% | -0.635 | 沈める | 浮かせる |
| L2 | SGD | full | switch_full | 85.9% | 72.3% | 0.753 | 沈める | 沈める |
| L2 | SGD | own | fit_full | 47.4% | 47.9% | 0.021 | 浮かせる | 沈める |
| L2 | SGD | own | switch_full | 71.8% | 71.3% | 0.614 | 浮かせる | 浮かせる |
| L2 | SGD | upstream | fit_full | 23.9% | 36.4% | -0.635 | 沈める | 浮かせる |
| L2 | SGD | upstream | switch_full | 85.1% | 70.7% | 0.748 | 沈める | 沈める |
| L3 | Adam_fixed | full | fit_full | 46.8% | 50.4% | -0.044 | 浮かせる | 沈める |
| L3 | Adam_fixed | full | switch_full | 99.1% | 99.5% | 0.882 | 浮かせる | 浮かせる |
| L3 | Adam_fixed | own | fit_full | 47.7% | 49.7% | -0.011 | 浮かせる | 沈める |
| L3 | Adam_fixed | own | switch_full | 85.5% | 51.2% | 0.717 | 浮かせる | 浮かせる |
| L3 | Adam_fixed | upstream | fit_full | 47.3% | 50.9% | -0.048 | 浮かせる | 沈める |
| L3 | Adam_fixed | upstream | switch_full | 99.3% | 99.6% | 0.899 | 浮かせる | 浮かせる |
| L3 | SGD | full | fit_full | 15.7% | 40.1% | -0.504 | 浮かせる | 沈める |
| L3 | SGD | full | switch_full | 98.9% | 91.9% | 0.766 | 浮かせる | 浮かせる |
| L3 | SGD | own | fit_full | 48.4% | 49.7% | -0.023 | 浮かせる | 沈める |
| L3 | SGD | own | switch_full | 83.9% | 51.0% | 0.830 | 浮かせる | 浮かせる |
| L3 | SGD | upstream | fit_full | 15.0% | 39.3% | -0.504 | 浮かせる | 沈める |
| L3 | SGD | upstream | switch_full | 99.1% | 94.5% | 0.768 | 浮かせる | 浮かせる |

## 状態の推移（SGD等と無関係な前活性）

| 腕 | 課題 | 更新 | 層 | r中央値 | s中央値 | 応答ありunit | 平均absゲート |
|---|---|---|---|---|---|---|---|
| SiLU | 1 | 10000 | L2 | -0.875 | 1.863 | 512 | 0.2494 |
| SiLU | 1 | 10000 | L3 | -0.740 | 2.169 | 512 | 0.2818 |
| SiLU | 5 | 10000 | L2 | -1.804 | 3.997 | 512 | 0.0823 |
| SiLU | 5 | 10000 | L3 | -2.900 | 4.190 | 512 | 0.0203 |
| SiLU | 10 | 10000 | L2 | -1.042 | 6.586 | 512 | 0.1790 |
| SiLU | 10 | 10000 | L3 | -1.038 | 5.214 | 512 | 0.1912 |
| SiLU | 25 | 10000 | L2 | -1.102 | 11.630 | 512 | 0.1475 |
| SiLU | 25 | 10000 | L3 | 0.625 | 8.707 | 512 | 0.6569 |
| SiLU | 50 | 10000 | L2 | -0.603 | 13.199 | 512 | 0.2907 |
| SiLU | 50 | 10000 | L3 | 1.792 | 14.818 | 463 | 0.7778 |
| SiLU | 100 | 100 | L2 | -2.685 | 17.383 | 429 | 0.1947 |
| SiLU | 100 | 100 | L3 | 4.412 | 14.220 | 431 | 0.8326 |
| SiLU | 100 | 10000 | L2 | -2.388 | 18.871 | 489 | 0.1860 |
| SiLU | 100 | 10000 | L3 | 2.230 | 21.983 | 440 | 0.8106 |
| SiLUcap | 10 | 10000 | L2 | -0.923 | 1.880 | 512 | 0.2330 |
| SiLUcap | 10 | 10000 | L3 | -2.529 | 2.194 | 512 | 0.0420 |
| SiLUcap | 25 | 10000 | L2 | -0.852 | 1.944 | 512 | 0.2483 |
| SiLUcap | 25 | 10000 | L3 | -3.744 | 2.229 | 512 | 0.0280 |
| SiLUcap | 50 | 10000 | L2 | -0.867 | 2.073 | 512 | 0.2461 |
| SiLUcap | 50 | 10000 | L3 | -4.225 | 2.281 | 512 | 0.0286 |
| SiLUcap | 100 | 10000 | L2 | -0.821 | 2.153 | 512 | 0.2586 |
| SiLUcap | 100 | 10000 | L3 | -5.184 | 2.288 | 512 | 0.0421 |

## 対照・数値感度

| 腕 | 層 | 分母 | head模型 | 実勾配経路 | BA 課題50末/100末 |
|---|---|---|---|---|---|
| SiLU | 2 | Adam_fixed | P2 | fit_body | 50.5% / 38.0% |
| SiLU | 2 | Adam_fixed | P2 | fit_full | 50.5% / 38.0% |
| SiLU | 2 | Adam_fixed | P2 | switch_body | 52.1% / 86.9% |
| SiLU | 2 | Adam_fixed | P2 | switch_full | 52.1% / 86.9% |
| SiLU | 2 | Adam_fixed | P2ctx | fit_body | 50.6% / 38.2% |
| SiLU | 2 | Adam_fixed | P2ctx | fit_full | 50.6% / 38.2% |
| SiLU | 2 | Adam_fixed | P2ctx | switch_body | 51.4% / 87.4% |
| SiLU | 2 | Adam_fixed | P2ctx | switch_full | 51.4% / 87.4% |
| SiLU | 2 | SGD | P2 | fit_body | 44.0% / 35.1% |
| SiLU | 2 | SGD | P2 | fit_full | 44.0% / 35.1% |
| SiLU | 2 | SGD | P2 | switch_body | 59.5% / 72.3% |
| SiLU | 2 | SGD | P2 | switch_full | 59.5% / 72.3% |
| SiLU | 2 | SGD | P2ctx | fit_body | 43.6% / 35.2% |
| SiLU | 2 | SGD | P2ctx | fit_full | 43.6% / 35.2% |
| SiLU | 2 | SGD | P2ctx | switch_body | 57.7% / 72.7% |
| SiLU | 2 | SGD | P2ctx | switch_full | 57.7% / 72.7% |
| SiLU | 3 | Adam_fixed | P2 | fit_body | 46.7% / 50.4% |
| SiLU | 3 | Adam_fixed | P2 | fit_full | 46.6% / 50.4% |
| SiLU | 3 | Adam_fixed | P2 | switch_body | 90.0% / 99.5% |
| SiLU | 3 | Adam_fixed | P2 | switch_full | 90.0% / 99.5% |
| SiLU | 3 | Adam_fixed | P2ctx | fit_body | 46.7% / 50.4% |
| SiLU | 3 | Adam_fixed | P2ctx | fit_full | 46.6% / 50.4% |
| SiLU | 3 | Adam_fixed | P2ctx | switch_body | 90.0% / 99.5% |
| SiLU | 3 | Adam_fixed | P2ctx | switch_full | 90.0% / 99.5% |
| SiLU | 3 | SGD | P2 | fit_body | 39.0% / 40.1% |
| SiLU | 3 | SGD | P2 | fit_full | 39.1% / 40.1% |
| SiLU | 3 | SGD | P2 | switch_body | 86.2% / 91.9% |
| SiLU | 3 | SGD | P2 | switch_full | 86.2% / 91.9% |
| SiLU | 3 | SGD | P2ctx | fit_body | 38.8% / 39.6% |
| SiLU | 3 | SGD | P2ctx | fit_full | 38.9% / 39.6% |
| SiLU | 3 | SGD | P2ctx | switch_body | 86.0% / 94.5% |
| SiLU | 3 | SGD | P2ctx | switch_full | 86.0% / 94.5% |
| SiLUcap | 2 | Adam_fixed | P2 | fit_body | 24.0% / 55.5% |
| SiLUcap | 2 | Adam_fixed | P2 | fit_full | 24.1% / 55.5% |
| SiLUcap | 2 | Adam_fixed | P2 | switch_body | 61.0% / 52.1% |
| SiLUcap | 2 | Adam_fixed | P2 | switch_full | 61.0% / 52.1% |
| SiLUcap | 2 | Adam_fixed | P2ctx | fit_body | 24.0% / 56.1% |
| SiLUcap | 2 | Adam_fixed | P2ctx | fit_full | 24.1% / 56.1% |
| SiLUcap | 2 | Adam_fixed | P2ctx | switch_body | 61.4% / 52.9% |
| SiLUcap | 2 | Adam_fixed | P2ctx | switch_full | 61.4% / 52.9% |
| SiLUcap | 2 | SGD | P2 | fit_body | 32.9% / 58.3% |
| SiLUcap | 2 | SGD | P2 | fit_full | 32.9% / 58.3% |
| SiLUcap | 2 | SGD | P2 | switch_body | 61.4% / 51.1% |
| SiLUcap | 2 | SGD | P2 | switch_full | 61.4% / 51.1% |
| SiLUcap | 2 | SGD | P2ctx | fit_body | 31.8% / 58.6% |
| SiLUcap | 2 | SGD | P2ctx | fit_full | 31.8% / 58.6% |
| SiLUcap | 2 | SGD | P2ctx | switch_body | 62.3% / 50.8% |
| SiLUcap | 2 | SGD | P2ctx | switch_full | 62.3% / 50.8% |
| SiLUcap | 3 | Adam_fixed | P2 | fit_body | 42.1% / 43.5% |
| SiLUcap | 3 | Adam_fixed | P2 | fit_full | 42.1% / 43.5% |
| SiLUcap | 3 | Adam_fixed | P2 | switch_body | 50.2% / 52.1% |
| SiLUcap | 3 | Adam_fixed | P2 | switch_full | 50.3% / 52.1% |
| SiLUcap | 3 | Adam_fixed | P2ctx | fit_body | 42.4% / 44.4% |
| SiLUcap | 3 | Adam_fixed | P2ctx | fit_full | 42.4% / 44.4% |
| SiLUcap | 3 | Adam_fixed | P2ctx | switch_body | 49.8% / 51.9% |
| SiLUcap | 3 | Adam_fixed | P2ctx | switch_full | 50.0% / 51.9% |
| SiLUcap | 3 | SGD | P2 | fit_body | 44.4% / 44.8% |
| SiLUcap | 3 | SGD | P2 | fit_full | 44.4% / 44.8% |
| SiLUcap | 3 | SGD | P2 | switch_body | 42.6% / 未定義 |
| SiLUcap | 3 | SGD | P2 | switch_full | 42.6% / 未定義 |
| SiLUcap | 3 | SGD | P2ctx | fit_body | 45.6% / 45.3% |
| SiLUcap | 3 | SGD | P2ctx | fit_full | 45.6% / 45.3% |
| SiLUcap | 3 | SGD | P2ctx | switch_body | 47.1% / 未定義 |
| SiLUcap | 3 | SGD | P2ctx | switch_full | 47.1% / 未定義 |

数値ゼロ閾値1e−8/1e−10/1e−12で対象数またはBAが変わる比較は104。詳細zero_sensitivity.json。固有値閾値でrankが変わった状態はP2tolを同じ表へ全て追加。

## 検査・限定

課題10/100末で、全パラメータのr方向を有限差分と照合。相2自体も半log擬似detの有限差分で検査。gate.jsonに刻み別の誤差を保存。全状態で保存layer_statsのr/sを照合し、body戻し＋共有headの直接勾配＝実CEの合計を検算。課題100更新100は元ログに同一保存点の比較行がないためreference_available=falseを明記。
相2模型では自由headをfitするが、実網はheadとembeddingを共有する。switch_body/fit_bodyは直接head更新を除いた対照であり、fullと取り違えない。
切替の独立新ラベル平均と現課題fitを分けた。保存点の一歩が一致しても、Adam運動量・課題の胴・幅成長を通した長期の沈降死の因果説明にはならない。L3の正側化が必ず起きるという主張でもない。
SiLUcap対照のAdam分母固定は、capの更新後射影を加える前の勾配方向。capの実更新そのものではなく、救済の媒介率とは読まない。

## 事前予測の採点

- SWITCH_MATCH: 予測0.50、観測False、Brier=0.250。
- FIT_MATCH: 予測0.20、観測False、Brier=0.040。
- LATE_SINK: 予測0.20、観測False、Brier=0.040。

## 置き場

spec登録87cbfd4a、実装62d059d8。code/results: chargpt_siluphase2_1024。生配列・ログはbackup_manifest.jsonを参照。入力checkpointは他worktreeから読取のみ。

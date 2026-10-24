# direct 成分の発生源の分解（探索的な第二段階）

第一段階の結果を見てから spec §7 を登録し、測定した。元の判定は変更しない。
self は自 unit の forward 出力、other は他 FFN unit の forward 出力、residual は固定残差入力が作る誤差信号。上流の更新ではない。
direct = self + other + residual が厳密に成立。最終 LN の分散微分はこの表には含めない。
各値は seed 内の窓中央値の seed 間の範囲。大きさの分担は絶対値で正規化した帳簿であり、因果的な寄与率ではない。

## S2

| 量 | 早期 | 中期 | 後期 |
|---|---|---|---|
| s_original_self | 0.9979–1.0000 | 0.3959–1.0000 | 0.0371–0.8899 |
| s_original_other | 0.4297–0.7981 | 0.1006–0.5593 | 0.2349–0.9551 |
| s_original_residual | 0.3200–0.4662 | 0.4447–0.5933 | 0.4345–0.5111 |
| s_original_direct | 0.9570–0.9902 | 0.5049–0.9314 | 0.4922–0.6902 |
| s_original_self_plus_other | 0.9752–0.9941 | 0.5234–0.9095 | 0.4990–0.6655 |
| s_original_self_plus_residual | 0.9771–0.9980 | 0.4473–1.0000 | 0.0400–0.8368 |
| original_self_abs_share | 0.5065–0.6368 | 0.1156–0.5824 | 0.1904–0.4725 |
| original_other_abs_share | 0.1756–0.2237 | 0.2521–0.5815 | 0.4199–0.6843 |
| original_residual_abs_share | 0.1232–0.2001 | 0.0825–0.2538 | 0.0785–0.1888 |
| original_self_positive_direct_nonpositive | 0.0098–0.0381 | 0.0686–0.2259 | 0.0098–0.3206 |
| s_loaderror_self | 0.9979–1.0000 | 0.4467–1.0000 | 0.0400–0.9085 |
| s_loaderror_other | 0.5479–0.7532 | 0.3109–0.6788 | 0.4046–0.6204 |
| s_loaderror_residual | 0.2852–0.3672 | 0.4628–0.5167 | 0.4293–0.5079 |
| s_loaderror_direct | 0.7168–0.8820 | 0.4671–0.7803 | 0.4805–0.6509 |
| s_loaderror_self_plus_other | 0.7227–0.9192 | 0.5049–0.7906 | 0.4766–0.6519 |
| s_loaderror_self_plus_residual | 0.6505–0.9414 | 0.4945–0.9961 | 0.0859–0.6736 |
| loaderror_self_abs_share | 0.3048–0.4911 | 0.0745–0.5216 | 0.0386–0.2723 |
| loaderror_other_abs_share | 0.2851–0.5313 | 0.2253–0.6793 | 0.4621–0.9037 |
| loaderror_residual_abs_share | 0.1756–0.2716 | 0.1061–0.3421 | 0.0323–0.4170 |
| loaderror_self_positive_direct_nonpositive | 0.1035–0.2832 | 0.1020–0.3402 | 0.0146–0.3560 |

## 検算

第一段階 direct との最大相対誤差 5.68e-15、softmax 恒等式 2.31e-14、token ごとの加法分解 9.92e-15。
load-only error + abs(K) の self 非負性も全状態で確認。

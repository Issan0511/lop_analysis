# direct 成分の発生源の分解（探索的な第二段階）

第一段階の結果を見てから spec §7 を登録し、測定した。元の判定は変更しない。
self は自 unit の forward 出力、other は他 FFN unit の forward 出力、residual は固定残差入力が作る誤差信号。上流の更新ではない。
direct = self + other + residual が厳密に成立。最終 LN の分散微分はこの表には含めない。
各値は seed 内の窓中央値の seed 間の範囲。大きさの分担は絶対値で正規化した帳簿であり、因果的な寄与率ではない。

## A2

| 量 | 早期 | 中期 | 後期 |
|---|---|---|---|
| s_original_self | 0.9975–1.0000 | 0.9364–1.0000 | 0.7500–1.0000 |
| s_original_other | 0.7553–0.8951 | 0.0244–0.6997 | 0.0059–0.5121 |
| s_original_residual | 0.6867–0.8979 | 0.4834–0.6890 | 0.4756–0.5068 |
| s_original_direct | 0.9864–0.9956 | 0.6641–0.8679 | 0.4932–0.7489 |
| s_original_self_plus_other | 0.9876–0.9959 | 0.6670–0.8959 | 0.5000–0.7044 |
| s_original_self_plus_residual | 0.9887–0.9958 | 0.8233–1.0000 | 0.6754–1.0000 |
| original_self_abs_share | 0.4761–0.5839 | 0.3152–0.5048 | 0.1867–0.4956 |
| original_other_abs_share | 0.1690–0.2021 | 0.1991–0.4712 | 0.3949–0.5896 |
| original_residual_abs_share | 0.1831–0.3111 | 0.0262–0.2678 | 0.0209–0.1789 |
| original_self_positive_direct_nonpositive | 0.0041–0.0086 | 0.1033–0.3359 | 0.1517–0.5068 |
| s_loaderror_self | 0.9975–1.0000 | 0.9366–1.0000 | 0.7473–1.0000 |
| s_loaderror_other | 0.5949–0.7654 | 0.1904–0.6667 | 0.0186–0.5284 |
| s_loaderror_residual | 0.4191–0.7208 | 0.4941–0.5999 | 0.4378–0.5082 |
| s_loaderror_direct | 0.8945–0.9669 | 0.5654–0.8166 | 0.5059–0.6287 |
| s_loaderror_self_plus_other | 0.9398–0.9854 | 0.5732–0.9069 | 0.5103–0.6578 |
| s_loaderror_self_plus_residual | 0.8937–0.9512 | 0.7520–1.0000 | 0.5858–1.0000 |
| loaderror_self_abs_share | 0.4948–0.5411 | 0.2872–0.4914 | 0.2006–0.4806 |
| loaderror_other_abs_share | 0.1513–0.3093 | 0.1318–0.4707 | 0.4566–0.5285 |
| loaderror_residual_abs_share | 0.2194–0.3262 | 0.0363–0.3520 | 0.0228–0.2022 |
| loaderror_self_positive_direct_nonpositive | 0.0290–0.1055 | 0.1308–0.4346 | 0.2420–0.4941 |

## 検算

第一段階 direct との最大相対誤差 2.6e-15、softmax 恒等式 1.62e-14、token ごとの加法分解 6.65e-15。
load-only error + abs(K) の self 非負性も全状態で確認。

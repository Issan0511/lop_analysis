# direct 成分の発生源の分解（探索的な第二段階）

第一段階の結果を見てから spec §7 を登録し、測定した。元の判定は変更しない。
self は自 unit の forward 出力、other は他 FFN unit の forward 出力、residual は固定残差入力が作る誤差信号。上流の更新ではない。
direct = self + other + residual が厳密に成立。最終 LN の分散微分はこの表には含めない。
各値は seed 内の窓中央値の seed 間の範囲。大きさの分担は絶対値で正規化した帳簿であり、因果的な寄与率ではない。

## A3

| 量 | 早期 | 中期 | 後期 |
|---|---|---|---|
| s_original_self | 0.9955–1.0000 | 0.9645–1.0000 | 0.9268–1.0000 |
| s_original_other | 0.6125–0.8879 | 0.5713–0.6250 | 0.3389–0.4893 |
| s_original_residual | 0.4629–0.6496 | 0.4404–0.4912 | 0.4639–0.4925 |
| s_original_direct | 0.7168–0.9939 | 0.6768–0.9395 | 0.6221–0.8721 |
| s_original_self_plus_other | 0.7515–0.9902 | 0.6826–0.9463 | 0.6165–0.8691 |
| s_original_self_plus_residual | 0.6973–0.9902 | 0.8330–1.0000 | 0.8369–1.0000 |
| original_self_abs_share | 0.1520–0.5121 | 0.2715–0.5818 | 0.1950–0.6197 |
| original_other_abs_share | 0.3532–0.4446 | 0.3288–0.5497 | 0.3145–0.5705 |
| original_residual_abs_share | 0.1008–0.4051 | 0.0615–0.1245 | 0.0561–0.1158 |
| original_self_positive_direct_nonpositive | 0.0041–0.2391 | 0.0605–0.2628 | 0.1279–0.3333 |
| s_loaderror_self | 0.9978–1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| s_loaderror_other | 0.5762–0.7227 | 0.5117–0.5400 | 0.4746–0.5322 |
| s_loaderror_residual | 0.4216–0.5759 | 0.4629–0.5000 | 0.4746–0.5088 |
| s_loaderror_direct | 0.6426–0.8480 | 0.5794–0.6445 | 0.5225–0.6133 |
| s_loaderror_self_plus_other | 0.6716–0.8551 | 0.5794–0.6455 | 0.5205–0.6123 |
| s_loaderror_self_plus_residual | 0.6176–0.9216 | 0.8301–0.9795 | 0.8121–0.9961 |
| loaderror_self_abs_share | 0.0679–0.3493 | 0.0877–0.2654 | 0.0578–0.3048 |
| loaderror_other_abs_share | 0.4126–0.5508 | 0.6227–0.8273 | 0.6376–0.8635 |
| loaderror_residual_abs_share | 0.1332–0.5817 | 0.0470–0.1126 | 0.0444–0.0805 |
| loaderror_self_positive_direct_nonpositive | 0.1449–0.3574 | 0.3555–0.4206 | 0.3867–0.4775 |

## 検算

第一段階 direct との最大相対誤差 1.27e-15、softmax 恒等式 1.17e-14、token ごとの加法分解 8.92e-15。
load-only error + abs(K) の self 非負性も全状態で確認。

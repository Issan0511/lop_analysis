# direct 成分の発生源の分解（探索的な第二段階）

第一段階の結果を見てから spec §7 を登録し、測定した。元の判定は変更しない。
self は自 unit の forward 出力、other は他 FFN unit の forward 出力、residual は固定残差入力が作る誤差信号。上流の更新ではない。
direct = self + other + residual が厳密に成立。最終 LN の分散微分はこの表には含めない。
各値は seed 内の窓中央値の seed 間の範囲。大きさの分担は絶対値で正規化した帳簿であり、因果的な寄与率ではない。

## A4

| 量 | 早期 | 中期 | 後期 |
|---|---|---|---|
| s_original_self | 0.9957–1.0000 | 0.9894–1.0000 | 0.9852–1.0000 |
| s_original_other | 0.6502–0.7786 | 0.5752–0.6510 | 0.5348–0.6032 |
| s_original_residual | 0.4675–0.6433 | 0.4720–0.5011 | 0.4579–0.4961 |
| s_original_direct | 0.7157–0.8287 | 0.6366–0.7314 | 0.5899–0.7043 |
| s_original_self_plus_other | 0.6850–0.8425 | 0.6462–0.7659 | 0.6055–0.7008 |
| s_original_self_plus_residual | 0.5304–0.7036 | 0.5629–0.6865 | 0.6461–0.7146 |
| original_self_abs_share | 0.0242–0.0893 | 0.0675–0.1849 | 0.0966–0.2123 |
| original_other_abs_share | 0.4158–0.5551 | 0.4838–0.6129 | 0.4940–0.7268 |
| original_residual_abs_share | 0.2956–0.4239 | 0.2396–0.3100 | 0.1550–0.2518 |
| original_self_positive_direct_nonpositive | 0.1699–0.2788 | 0.2583–0.3585 | 0.2957–0.3879 |
| s_loaderror_self | 1.0000–1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| s_loaderror_other | 0.6028–0.7235 | 0.5518–0.5904 | 0.5299–0.5733 |
| s_loaderror_residual | 0.4872–0.6336 | 0.4701–0.5171 | 0.4565–0.4971 |
| s_loaderror_direct | 0.6353–0.7214 | 0.5618–0.6187 | 0.5562–0.5998 |
| s_loaderror_self_plus_other | 0.6314–0.7500 | 0.5756–0.6337 | 0.5593–0.6150 |
| s_loaderror_self_plus_residual | 0.5020–0.6455 | 0.5510–0.6304 | 0.6069–0.6768 |
| loaderror_self_abs_share | 0.0143–0.0565 | 0.0373–0.1039 | 0.0562–0.1237 |
| loaderror_other_abs_share | 0.4342–0.6168 | 0.5910–0.7090 | 0.6104–0.8265 |
| loaderror_residual_abs_share | 0.2919–0.5015 | 0.2108–0.3099 | 0.1171–0.2334 |
| loaderror_self_positive_direct_nonpositive | 0.2786–0.3647 | 0.3813–0.4382 | 0.4002–0.4438 |

## 検算

第一段階 direct との最大相対誤差 1.1e-15、softmax 恒等式 7.1e-15、token ごとの加法分解 9.85e-15。
load-only error + abs(K) の self 非負性も全状態で確認。

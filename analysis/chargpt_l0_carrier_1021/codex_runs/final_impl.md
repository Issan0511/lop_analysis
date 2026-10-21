(A)(B) の計算を完了し、[summary.md](/home/issan/Projects/claude/wt/chargpt_l0_carrier_1021/results/chargpt_l0_carrier_1021/summary.md) に保存しました。検査は全通過し、恒等式の最大残差は **1.74×10⁻¹³** でした。commit はしていません。

| 予測 | 採点 | 主な結果 |
|---|---|---|
| P-A1 | ✗ | 上流割合の中央値 0.322。条件成立 0/10 |
| P-A2 | ✗ | 変位の伸び条件は成立。入力変化が腕間1.5倍以内という条件は不成立 |
| P-A3 | ✗ | 自分の項の cap−対照は +0.124 |
| P-A4 | ✗ | cap の回転量増加は10/10。比の中央値 21.88 |
| P-A5 | 判定不能 | 各課題末で判定すると✗。課題を中央値で集約すると○ |
| P-B1 | ○ | 序盤のFFN0割合の条件は両腕10/10 |
| P-B2 | ✗ | 終盤の条件はC3で7/10、capで0/10 |

上流項の尺度差と、自分の項・回転の差の両方が現れました。固定状態の分解なので、救済の因果経路は特定していません。

指定CSV・圧縮unit配列・[checks.json](/home/issan/Projects/claude/wt/chargpt_l0_carrier_1021/results/chargpt_l0_carrier_1021/checks.json) も保存済みです。spec と (C) は変更していません。
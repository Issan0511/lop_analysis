# T5複製実験の途中結果

完了した条件のみ: 1/80。各条件32複製とfull-batch。未完了条件を0として集計しない。

入力とfc1以外を固定した局所介入。μ・rはunit平均、標本抽出ありはさらに32複製で平均する。
雑音効果は「複製平均−full-batch」、SEは固定した1初期状態での複製間Monte Carlo標準誤差。seed間の再現性ではない。
次のゲート共分散項は共通固定計量での期待Δμであり、有限50更新の寄与率でも、実Adamの変位分解でもない。

| seed | 課題 | 起点 | block | optimizer | full Δμ | 複製 Δμ | 雑音効果 ± SE | 次の共分散項 | 次の全体 |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 10 | 100 | 1 | fixed | -0.00106908 | -0.001062685 | 6.39e-06 ± 4.93e-06 | -1.475e-08 | -2.06e-05 |

全時点とrの変化は[t5_completed_cases.csv](t5_completed_cases.csv)。
解析コード・初期checkpoint・生の複製・検算は各directoryのprovenance.json、manifest.json、complete.jsonから追跡できる。

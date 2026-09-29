# E残差replayのfloat64精度対照

事後の検算としてcommit 616cbefで実行前に登録。元のfloat32主行列は変更していない。新しいseed・horizon・機構腕は追加せず、既存の両seedのtask50 checkpointから、reference c=1とE c=0.1/10のみをtask51–53の同一planで12000歩ずつ実行した。合計約44.5秒。

## 結果

**数学的な尺度等価性に対して、float64では全unitの差が小さくなった。厳密な全軌道bit一致は達成していないが、float32主行列で見られた大きな少数unitのずれはこのshadowでは出なかった。**

task53終了時のreferenceとの差：

| seed | gain | 最大abs ΔT | median abs ΔT | hidden相対L2誤差 | abs ΔT>1e−6のunit数 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.1 | 7.065e−6 | 1.235e−13 | 9.316e−8 | 2 |
| 0 | 10 | 9.152e−6 | 1.166e−13 | 1.000e−7 | 1 |
| 1 | 0.1 | 8.618e−8 | 1.841e−13 | 2.878e−9 | 0 |
| 1 | 10 | 1.124e−7 | 1.632e−13 | 2.210e−9 | 0 |

- 全診断点にわたる最大abs ΔTは9.152e−6、median abs ΔTの最大は1.841e−13。
- abs ΔT>1e−4、>1e−2、>0.1のunitは、全seed・arm・診断点で0。
- task51末では最大abs ΔT≤2.345e−10、task52末では≤9.848e−9。差は時間とともに増幅し得るが、この12000歩では上表の範囲にとどまった。
- hiddenの全座標をまとめた相対L2誤差は最大1.001e−7。最大座標差は3.376e−6。
- b2は全診断点でreferenceとbit一致。パラメータ、Adam moments、Tはfiniteで、s>0。
- 読み出し値は各armで最後まで凍結されていることを確認。

## 独立した同一状態からの一歩

開始時と各task末に、現在のreferenceのP/M/Sを新たにコピーし、gain0.1/10をdouble精度で変更したEを構成した。その同じ状態・同じバッチからreferenceとEを一歩だけ進める比較を行った。これは長期に分岐したEの状態を使う比較ではない。

- 次のhidden parameterの最大差：4.441e−16。
- その差／referenceの最大座標更新量：最大1.182e−13。
- b2の差：0。

従って、同じ状態からの尺度等価式と実装はdouble精度の丸めの範囲で整合する。一方、同じ状態からの一歩の一致は、摂動に対する長時間安定性の証明ではない。

## 何を検算しているか

元task50に保存されたfloat32のP/M/Sを**先にdoubleへ移し、その後に**読み出しをa倍、hidden M/Sをa/a²倍する。hidden epsilonはa倍、出力biasのmoment/epsilonは変更しない。200000歩まで進んだoptimizer counterを保持する。

referenceの同じ一歩のnative CE logits gradient（batch平均の除算済）をEのbackwardに注入し、hiddenとb2に使う。完全な同一状態・厳密算術なら、hidden gradientとmomentはa/a²倍の関係を保ち、hidden更新はreferenceと同じになる。b2はそのまま同じ勾配・momentを使う。

## 限界

これは**元float32軌道の再現ではなく、尺度等価式の精度検算**である。ELU backwardは同じ式expm1(z)+1でも、float64化すると丸めによる零化位置が変わるため、reference自体のその後の軌道がfloat32とは異なる。従って元の少数outlierの分岐点や原因を一つずつ特定したわけではない。

残差replayのreference軌道は厳密算術では解だが、横方向の摂動に安定とは限らない。残差を外から固定したhidden勾配の局所曲率には、残差で重み付けされたELU曲率が残る一方、通常のCEにある正半定値の残差フィードバック項JᵀCJはなくなる。これは丸め差増幅の候補であり、本検算はその安定性解析までは行っていない。

主float32の中央値が零に近いことと、全unitの全軌道が一致することは区別する。本shadowは尺度等価式・実装との整合を支持するが、Eの零予測自体を新しい機構の発見として数えない。追加の精度上昇やhorizon延長は行わない。

## ファイル

- real_null_precision_config.json：実行前に固定した設定。
- real_null_precision_summary.csv：全診断点の最大差・中央値・outlier数・相対誤差。
- real_null_precision_s0_units.csv / real_null_precision_s1_units.csv：各unitの値。
- real_null_precision_same_state.csv：同じreference状態からの独立一歩検算。
- real_null_precision_s0_final_states.npz / real_null_precision_s1_final_states.npz：各armの最終parameterとmoments。
- real_null_precision_s0_provenance.json / real_null_precision_s1_provenance.json：元checkpoint・task plan・実行codeのSHA256。
- 実行コード：analysis/logistic_v_height_0929/real_null_precision.py。

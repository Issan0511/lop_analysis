# 相2の方向を保存済みの押しで示す（1024）

これは chargpt_l0phase2_1023 の既存配列を図にする追補で、新たな機構判定・学習・checkpoint の変更は行わない。

- 予測: 最終headの全方向を共同最小二乗fitする条件付き相2模型、iid旧ラベル共分散。`headcov_*` の `SGD_iid` / `Adam_fixed_iid`。
- 実測: 同じ保存状態・同じunitへの独立新課題ラベル平均CEの押し（`p - 1/65`）。入力は現課題の鎖probe。`scalar_*` の `*_switch_r`。
- 押し S > 0 は r が下がる向き。L0 up行だけを更新したときの r の一次変化が -ηS。自己項だけを取り出さず、他unit、後段注意・FFN・LNを含む勾配を使う。Adamは保存分母固定、運動量を含む実更新そのものではない。
- 主図: C3・課題100末・Adam固定分母。全10seed、固定一様probeで生きている全unit。各seed内で予測値の昇順に並べ、同じunitの予測と実測を2本の帯で対応させる。予測で沈む群と浮く群それぞれの、実測で沈む割合も示す。ReLUcap・SGDの図も出す。
- ゼロは別の色。二方向の符号一致とbalanced accuracyは予測・実測とも非ゼロのunitで計算する。これは以前の `>0 / <=0` の2値判定と別の集計であり、旧集計も再現して残す。ゼロの閾値は厳密な0。閾値選択や係数fitは行わない。
- 表は両腕・全10seed・全5保存状態（9末、39末、99末、100の100更新、100末）の利用可能なSGD/Adamを含める。overviewは二方向balanced accuracyを全seed表示する。balanced accuracy = 実測で沈む群の的中率と実測で浮く群の的中率の平均。
- 独立旧ラベル平均の間Hを固定し、headは自由にfitできる条件付き模型。予測の絶対係数は未校正。全パラメータ更新や長期のL0沈降を示す図とはしない。

再生成:
```bash
MPLCONFIGDIR=/tmp/chargpt_phase2_mpl /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_phase2_direction_1024/plot.py
```

元データは `~/Projects/obsidian-research-data/chargpt_l0phase2_1023/results/chargpt_l0phase2_1023/`。コードと数値の親commitは `6f780068e043520a50bb4b3e5c303420ed88927e`。使用ファイルのSHA、旧集計との一致検査はprovenance.jsonへ記録する。

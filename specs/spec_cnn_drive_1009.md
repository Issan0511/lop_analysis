# CNN の駆動源の符号 — 証明と反例の検算（1009）

依頼: CNN で、循環的でない非空な仮定の下で、他チャネルを含む駆動が自己項と同じ向きになることを厳密に示す。

これは新しい学習実験の事前登録ではない。数学的な式・反例・十分条件の検算であり、既存 CNN の観測結果の再現や、実際の Adam 軌道での成立率を判定するものではない。

## 完了条件

1. 共有フィルタ、画像内の複数 Pool 領域、画像単位のラベルを式に含める。
2. bias シフトと平均パッチ方向を区別する。
3. 自己形・他チャネルを消した自己モデル・全体の項を区別する。
4. 仮定は符号そのものの言い換えにしない。実現可能な画像と CNN で条件が厳密な余裕を持って成立する例を示す。
5. 仮定を外すと反転する、実現可能な CNN の反例を示す。
6. 容量微分と学習後の実更新を混同しない。有限リッジ、CE、MaxPool の折れ目、深い網、Adam、時間平均への範囲を明記する。
7. 行列式・連鎖律・ラベル全列挙を独立に照合する。

## 今回検算する命題

- 平均パッチ方向の局所微分と、MaxPool の選択位置を使った微分。
- hidden を固定して readout を ridge fit した後の、新ラベルでの勾配平均の厳密式。
- 全 readout を残した有限行列の条件による自己形の符号保証。
- 自己モデルで白色化した、他チャネルのカーネルのスペクトル条件による容量の符号一致。
- ゼロ head から旧ラベルで CE を一歩学習した後の、切替勾配の符号。
- 正の hidden 学習率による有限の全層共同学習と、CE head 多歩の短い有限区間への拡張。
- 反応画像群を分けた CNN で、通常ラベル再利用を許す長期の条件維持と累積沈降。出力 bias なし・二乗損失・exact ridge head・full-batch SGD に限定する。
- 深い CNN や全訓練期間にそのまま外挿できないこと。

## 保存

導出: `analysis/cnn_drive_1009/proof.md`。
検算: `analysis/cnn_drive_1009/verify.py`。
結果: `results/cnn_drive_1009/verification.json`、`summary.md`。

不足が残れば依頼全体を達成済みとは記録しない。

## 1009 複数画像の通常Adamへの数学的検算範囲

ReLUのまま、異なる正画像3枚・rank3のhidden特徴とlogit Jacobianを持つnative Conv5/FC/全bias構成へ拡張する。binary iid labelsを画像ごとに独立に引き、全画像のfull-batchを有限H回reuseする。課題間moments保持・power-decaying学習率は維持する。画像平均を保つ同level摂動、全8ラベル割当のraw勾配、固定ridgeにおける元のfull/self容量微分を有限照合する。vector contrastの終点写像、正のAdam対角係数で重み付けした初期mean方向、population riskのobservable平均化は解析対象であり、長期成功率を数値推定する実験ではない。有限verifierの任意正対角metricを定常Adam係数の測定値とは扱わない。

## 1009 追加の数学的検算範囲

重なる channel と full-rank な空間特徴を持つ、bias 無し 1×1 CNN の全 raw CE-SGD 共同更新について、fresh labels・対称 channel 初期化・指定の適応学習率下での条件維持、head 極限、長期 mean 低下、full/isolated 全 NTK の符号を検算する。有限の300更新は導出式の照合であり、無限時間の実験的判定ではない。

また、小振幅の固定 iid 勾配 family に対する定常 Adam の展開・有限残差・対称性による符号保証を検算する。実 RL-CIFAR の変化する学習軌道と取り違えない。

## 1009 通常Adamの課題反復・無限時間への数学的検算範囲

追加で、native5×5 Conv・hiddenFC・全biasを含むbinary CNNの、均等予測状態の近傍にある全raw空間の開集合を調べる。単一画像、課題ごとのiid binary label、有限H回reuse、moments保持、通常のpower-decaying scalar learning rateを明示する。等重み不変線を仮定せず、平均ODEの終点写像とobservable平均化から高確率の永久保持・strict net mean下降を証明できるか検討する。Jacobian/NTKとCE勾配の有限照合は導出検算で、実RL-CIFARの成立率の実験ではない。

二段の共有Conv/ReLU/MaxPoolとbinary headを全raw共同更新し、Adamのmomentsを課題間で保持する。課題ごとのiid labelsを有限H回のfull-batch更新に再利用する。明示的な減衰・damping付きscalar learning rateの下で、全raw振幅が一致する不変線、固定状態の課題ブロック平均、実際の移動状態との確率的平均化誤差を扱う。任意に高い事前成功確率を持つ長期沈降の非空例が得られるかを解析し、通常一定学習率のRL-CIFARとの条件差を保持する。

有限のraw Adamと再帰・全NTK微分の照合は導出検算であり、無限時間の確率や実RL-CIFARでの成立率を数値推定する実験ではない。bias無し・binary・rank1の画像特徴という新しい構成の制限を明示する。ReLU以外へ拡張しない。

## 1009 課題内ラベル再利用の数学的検算範囲

同じラベル割当を課題内で固定する全raw共同更新について、指定の減衰・damping付きSGDの長期定理、一般CNNのfinite-task勾配誤差、negative mean/mixed gatesの非空例、過去Adam momentsを継続したfinite-task参照と外向き有理区間を検算する。これらは既存の式の検算であり、実RL-CIFARの統計的再実験ではない。

本来の5×5 sharedConvと全biasを持つ構造で、全rawNTKのfull/self方向微分と有限Jacobian摂動の式を検算する。任意有限本数のhiddenFCは解析的な同次性・rank-preserving構成で扱う。他の活性化関数には拡張しない。

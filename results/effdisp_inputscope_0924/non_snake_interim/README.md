# 実効変位と幅：ここまでの結果（0924・先行報告）

状態: **leaky限定60系列の集計完了。Snakeを含む追加130系列の全体報告は未完了。**
2026-09-24。数値正本は同フォルダの[summary.md](summary.md)と[verdict.csv](verdict.csv)。
この報告は作業中の出力で、訓練ソースのcommitを固定しているため結果commitは本走完了後に行う。

## 読む順番

1. [最初の環境・活性化横断検証（完了）](/home/issan/Projects/claude/proj_004_drift/results/effdisp_validation_0924/README.md)
2. この先行報告：入力変化を分けた再検証のleaky部分。
3. 数値・全条件・事後照合の詳細は[summary.md](summary.md)。

## 1. 入力を固定すれば幅は止まるか

Adam、leaky .1。CondAでは入力contextの固定/変更と教師側contextの固定/変更を分離。MNISTではpixel permutationの固定/変更とrandom labelsの固定/変更を分離した。
CondA40系列＋MNIST20系列、各条件5seed。入力と標的の両方を固定した条件なども全て集計した。
以下は**入力固定・標的変更**の条件。

| 環境 | 観測期間 | 近い相殺P1 | 頭打ちP2 |
| --- | --- | --- | --- |
| MNIST | 50tasks、末尾41–50 | 5/5 | 0/5 |
| MNIST | 150tasks、末尾121–150 | 5/5 | 0/5 |
| CondA、k1 | 400tasks、末尾321–400 | 0/5 | 0/5 |
| CondA、k7 | 400tasks、末尾321–400 | 1/5 | 1/5 |

群の通過基準は4/5。**この観測期間・optimizer・活性化では、入力固定だけで頭打ちにはならなかった。** 将来も収束しないという証明ではない。
MNISTでは供給と削りが近くても、残りが積み重なり幅は増える。前半の一定係数から後半を予測するP3も、表の条件では群として不成立。
「150tasksでは増加が減速する」という事前予測は外れた。二乗幅Vの末尾log傾き中央値は50tasksで.748、150tasksで1.192。

CondAの入力固定は課題を知らせるcontextも除くため、純粋な平均移動だけの介入ではない。

![MNISTの実効幅（二乗量V、seed中央値と範囲）](growth_all_mnist.png)

## 2. D/(2|c|)の上なのか、下なのか（質問に応じた事後解析）

R=√Vprev、D=√Q、c<0のtaskでRstar=D/(2|c|)。同じ更新を使えばR/Rstar=−2X/Qであり、近さ自体は独立な予測の証拠ではない。

| 条件 | 窓 | R/Rstarの中央値 |
| --- | --- | ---: |
| MNIST・固定入力 | task41–50 | .933 |
| MNIST・固定入力 | task121–150 | .957 |
| CondA・固定入力k1 | task321–400、c<0のtaskのみ | 1.550 |
| CondA・固定入力k7 | task321–400、c<0のtaskのみ | 1.864 |

task内中央値をseedごとに取り、5seedの中央値を表示。MNISTは全late taskでc<0。CondAは全体の約3割でc≥0となり、そのtaskにはこの式の正の釣り合い点はない。
CondAでは上側のtaskで実際に幅が縮み、別taskでの増加がそれを上回り得る。負のcのtaskだけの中央値と、全taskのQ加重収支比は別の統計量。
出典: `threshold_relation_posthoc.csv`・`threshold_relation_posthoc_groups.csv`。

## 3. 釣り合う高さ自体も動いていた（事後）

MNIST・固定入力leaky、task41–50 → 121–150の窓平均の対応する比、5seed中央値:

| 量 | 変化 |
| --- | ---: |
| D | +42% |
| cの絶対値 | −13% |
| D/(2 abs(c)) | +65% |
| 実効幅R | +68% |

入力分布が固定でも、更新量と更新方向の統計は一定になっていなかった。観測としては動く目安の近くで幅も増えている。
なぜDとcが変わるのかを、同じ更新の逆算から独立に説明・予測することが残る。
出典: `threshold_components_posthoc.csv`、comparison=`late50_to_late150`。

## 4. 元のRL-CIFARは収束していたか（元データの事後照合）

元の主張ノートが使った `sgd_postfit_cifar_0923/posthoc_c_eff.csv` を直接照合した。最初の検証で使った0918参照とは別の、**元の走そのものの保存CSV**である。

Adam A、10seed、task31–40 → 41–50の対応する窓平均比:

- 実効幅R: +8.3%。
- 実効変位D: +5.3%。
- D/(2|c|): +5.8%。

同じ頭打ち基準を事後適用するとP1は10/10、P2は0/10。late二乗幅Vのlog傾き中央値は.520。
**元のCIFARでも完全な停止は確認できず、近い相殺と成長の減速を示していた。** 元ノートの「伸びが止まっている」という表現はデータより強い。無限時間に有界でないという結論ではない。
出典: `original_rlcifar_threshold_posthoc.csv`・`original_rlcifar_plateau_posthoc.csv`・`original_rlcifar_posthoc_provenance.json`。

## 5. CondAのSGDはどうだったか

先行実験 `effdisp_validation_0924` の `k1_sgd`：leaky .1、SGD lr=.01、100tasks、毎task persistent1bit反転。

- late81–100でP1/P2は0/5。頭打ち未確認。
- c<0のtask割合のseed中央値は.6。約4割でc≥0。
- 実効幅の窓平均はtask61–80 → 81–100で+7.4%（対応比のseed中央値、事後）。
- lateの課題開始MSE中央値2.53 → 課題末.021。個々の課題の学習は進んでいる。

**入力固定のSGD factorialは未実施。** この追加走のoptimizerはAdamであり、SGDにも同じ入力介入を行ったとは言わない。
出典: 先行実験の`seed_summary.csv`・`verdict.csv`のconda/k1_sgd/effective、追加事後`prior_conda_sgd_threshold_posthoc.csv`。

## 6. 残っている比較

- Snake c=.6を中心とする入力/標的の比較、固定Snake α=.05とc=.3の対照：実行中。
- CIFAR画像のRGB32、64画素複製、複製/2、灰色32、平均縮小16 × Adam/SGD：実装・CPU検証済み、本走未開始。

最初の検証ではMNISTのSnake c=.3がP2を5/5で通ったが、αが下限.05にほぼ張り付いた。適応機構による効果と断定せず、上の対照で確かめる。

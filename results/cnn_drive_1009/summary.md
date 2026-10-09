# CNN 駆動源の条件付き定理と反例 — 1009 時点

状態: 条件付き定理・非空な構成・反例の検算済み。実 RL-CIFAR CNN の長期 CE/Adam 軌道を説明し切る依頼全体は未完了。

## 言えること

1. 共有 Conv、画像内の複数 MaxPool、画像単位のラベルを保ったまま、実際の平均前活性の SGD 変化を計算した。winner の勾配が共有重みを更新するため、非 winner の前活性も変わる。
2. 凍結した CNN の全線形 head を有限 ridge で当てはめた模型では、新ラベル切替勾配の期待値が厳密に求まる。自己形との角度と全特徴カーネルの固有値の条件で、他チャネルを残した期待沈降を保証した。ランク不足にも対応する。
3. 別に、容量の微分について、他ユニットを除いた自己モデルと全モデルが同符号となる有限強度の定理を示した。他ユニットを弱くする必要はなく、自己モデルで白色化した後の方向ごとの偏りを制限する。
4. 二乗損失・全 head ridge・共通学習率 SGD では、全 hidden 層を正の学習率で共同学習する有限期間へ拡張した。旧ラベルと学習後表現の依存を保ち、参照状態との差を一様評価する。二段 Conv の全旧ラベル検算も通過。
5. CE では、固定 hidden・ゼロ head からの一歩学習と、明示した正の総学習率範囲での多歩学習を証明した。長く head を学ぶと逆転する反例もある。
6. 反応する画像群を分けた三チャネルの共有 Conv→ReLU→MaxPool では、通常のラベル再利用を含めて長期沈降を証明した。出力 bias なし、exact ridge head、二乗損失、full-batch SGD という構成。指定の初期条件から、将来ラベルの乱数に関して少なくとも 12/13 の確率で条件領域を永久に保ち、その事象上で正の pooled 出力が 0 に収束する。平均前活性の累積変化は負、時間平均勾配の極限は 0。一般の CNN の成功確率ではない。
7. その後、任意の学習済み full CNN 状態で使える CE 条件へ広げた。全パラメータを通った logits の平均前活性方向感度と、クラス順位または top confidence/contrast を比較する。実 RL-CIFAR と同じ二段 Conv・二段 hidden FC・全 bias・重なる全 Conv チャネル・異なる二画像の異なる予測クラスを含む非空な開集合を解析的に構成した。旧 head の短い学習に限定しない。
8. 真の Adam 更新について、過去 momentum・座標ごとの倍率差・現在の勾配と分母の共分散を分けた有限状態の十分条件を示した。対称なクラス係数による厳密正例もあり、実 RL-CIFAR と同じ構造、10 クラス、batch 16、全 bias の非空例を検算した。対称性を少し崩した近傍も正になる。条件の実学習軌道での維持は未証明。
9. ユーザーの「一歩の Adam 反転は長期平均なら関係なくなるのでは」という指摘を受け、初期反転と定常偏りを分けた。一定の iid 勾配分布を Adam に供給する模型では、標準の beta と有限 epsilon でも、正の SGD 平均に対して負の定常 Adam 平均・ほぼ確実な時間平均が得られる。これは実 CNN のパラメータが変化する長期軌道の反転を証明したものではない。


10. 重なる複数チャネル、full-rank な空間特徴、自由な多クラス head を持ち、全 Conv/head が実際に変わる CE-SGD の無限時間過程で、最終平均前活性の strict な低下を証明した。初期 D=a²−||B||²>0 と指定の適応学習率から D>0、active ReLU、一意な MaxPool が永久に保たれ、head の画像上の成分が消える。a∞²≤a0²−||B0 P_U||²。1×1 Conv、bias 無し、channel 対称 family、毎更新 fresh iid labels が条件。a∞>0 なので ReLU の死は結論しない。
11. 上の二段 CNN の第一 Conv mean 方向では、全 raw parameter の NTK と、他の第一 Conv channel を除いた自己モデルの両方の容量微分を直接計算し、strict に同符号と示した。実 CE の期待方向はこれを反転しない。logits の class contrast が非零なら strict に一致、uniform 出力では CE 駆動が 0 となる。
12. 小振幅の定常 Adam で、正の平均信号 O(s³) と勾配雑音 O(s) の相対比較を有限残差つきで導出した。epsilon 支配だけでは方向は保証できない。一方 iid 雑音が厳密に正負対称なら任意の有限振幅で期待 Adam は正となり、epsilon 支配で相対的にも SGD の向き・大きさに近づく。負の平均前活性、重なる二チャネル、rank 2、10 classes、batch 16 の CNN 状態 family で実現した。固定状態の定常応答であり、更新中にその対称性が保たれるとは限らない。

13. 実際に状態が変わる Adam 軌道に対して、第一 Conv の累積下降量を期待勾配・martingale・分母の差・momentum の端点と重み変化へ厳密分解した。対称 CNN の構造と正振幅は指定の学習率上限で Adam 自体にも保存される。iid ラベルで残差が信号より小さいことは未証明。各画像に全 class を一回ずつ与える balanced-label batch という別条件なら、全 Conv/head の actual Adam 共同学習で mean が単調低下する非空な長期正例がある。


14. 同じラベルを課題内で固定してH回再利用する長期CE-SGDへ拡張した。課題境界で新ラベルを平均し、課題中のstate/label相関を総和有限なO(H²δ_k²)誤差として抑える。指定の減衰・damping付き学習率なら、元のstrict最終mean低下を維持する。H=30000とequal-coverageという実験スケジュールは含むが、一定学習率・Adamへの結論ではない。
15. 全raw SGDの有限課題では、負のmean（−0.1845）、正負のReLU sites、rank2の重なる非比例チャネル、全biasを持つ例まで保証した。全9ラベル割当を各8更新再利用し、期待下降下界9.59373196875e−6を解析的に証明。full/self容量微分も全パラメータ半径.005の球全体で正。
16. 通常Adamでも、課題境界でparameters・過去moments・global stepを保った参照と有限誤差から、課題内のラベル再利用を扱える。前課題のiidラベルを2更新した到達状態から、新しい全4割当を各3更新する例で、外向き有理区間による期待下降下界>1.976e−5。個別にmeanが上がる割当も含む。無限反復で条件が保たれることは未証明。
17. 本来の二段5×5 Conv、任意有限個のhidden FC/ReLU、全trainable biasを含め、元の全raw NTKとliteral自己モデルの容量微分が正となる開集合を構成した。参照からのJacobian距離の有限評価で、実状態の非比例チャネル・非零biasを許す。新ラベルCEの方向も別に一致を証明。学習軌道の条件維持は別問題。

## 反例と残る制限

非負入力・重み共有・一意な MaxPool・学習した出力 bias があっても、他チャネルによって自己モデルと実切替勾配の向きは逆転し得る。したがって無条件な一般化はできない。

容量の符号定理、有限 ridge 後の実勾配、CE の初期学習、全層共同学習、限定構成での長期沈降は、それぞれ仮定の異なる命題である。まとめて実 RL-CIFAR の定理とはしない。

実 RL-CIFAR の二段 Conv・二段非線形 FC・CE・batch 16・Adam について、重なるチャネルの競合、通常のラベル再利用、条件の軌道上での維持、Adam の座標依存更新を同時に扱う証明は未完。保存済みの集計 CSV だけでは今回の行列条件を復元・実測できない。

## 検算

- `verification.json`: 有理数の正例・反例、全ラベル列挙、共有 Conv の自動微分、複数 Pool 領域、ランク不足、CE の反転。
- `joint_verification.json`: 二段 Conv の共同学習、全旧ラベルでの誤差評価。正の学習率を許す非空性は解析的に証明し、具体的な小数値は float64 検算として扱う。
- `ce_interval_verification.json`: CE 多歩の有限区間と、小さなラベル依存表現変化を許す上界。
- `longtime_verification.json`: 10 クラス・9 画像、共有 Conv 自動微分 200 更新と再帰 50,000 更新。有限計算は再帰式を検算するもので、無限時間の主張は超マルチンゲールの証明に依る。

導出は `analysis/cnn_drive_1009/proof.md` と三つの付録。二つの独立最終監査を同じディレクトリに保存した。

## 追加検算と注意

- `ce_full_architecture.json`、`ce_different_classes.json`: 実 RL-CIFAR と同じ full architecture での、全パラメータ CE 恒等式・クラス順位条件・異なる予測 class・定義した CE self と full の符号。自動微分と解析式が一致。
- `adam_full_architecture.json`: full architecture、10 クラス、batch 16 のラベル全組合せを和の分布に集約。対称例で期待第一 Conv mean 変化は負、非対称例では期待 SGD 勾配が正でも最初の Adam mean 変化が正。真の torch.optim.Adam との誤差は 2.6e-15 以下。
- `adam_interval.json`、`adam_shared_conv.json`、`adam_symmetric.json`: 区間による分母誤差、有限履歴の非空例、全 class-count の共有 Conv 自動微分、対称ノイズの正例。
- `overlap_obstruction.json`: 非比例な重なりを持つ depthwise CNN の反例。任意小の overlap でも標的振幅が小さいと、旧ラベルを固定した方向が反転する。これは長期の非沈降を示すものではない。
- `adam_stationary_twopoint.json`: 16枚中1枚のみが標的勾配に寄与する、独立な10クラスラベルの二点分布。SGD平均 +6.25e-7、定常Adam期待の解析上界 -2.819430718e-5。Fraction による厳密検算。

追加のCE selfは、他の第一Convチャネルを取り去り forward を計算し直す明示した操作。元の logdet 自己項と同一の意味ではない。新しい十分条件を得ても、自己項の定義の違いを消したことにはならない。
- `adam_stationary_b16.json`: 16 枚全てが寄与する17値の固定 iid 勾配。SGD平均 +1e-6、定常Adam平均の厳密区間は約 [-0.0003586082662003, -0.0002715579654668]。有理数の moment と平方根の有理区間を用い、Taylor残差・低分母事象・epsilonを全て含む。独立な joint-moment 再帰でも照合。通常の初期化とbias correctionを含む時間平均も同じ極限へ収束する。固定分布への応答であり、実 CNN の変化するパラメータ軌道ではない。
- `adam_stationary_b16_optional_mc.json`: 別の補助シミュレーション出力も保存。符号判定はこの Monte Carlo の点推定によらず、上の有理数証明書による。


## 動き続ける CE-CNN と小振幅 Adam の追加検算

- `moving_ce_kernel.json`: rank 4、3 channels、10 classes、二段共有 Conv/ReLU/MaxPool の全 raw SGD を300更新。再帰との最大差1.34e-15未満。初期・更新後の full/self 全 NTK と方向微分の解析式との差3.56e-15未満。無限時間の結論は `moving_ce_longtime.md` の証明による。
- `adam_smallscale.json`: epsilon より小さい second moment でも cubic signal の向きが逆転する有限有理数上界、対称分布の相対誤差上界、第三 moment が0だけでは足りない反例。
- `adam_smallscale_cnn.json`: 負の平均前活性を持つ shared CNN で signal/noise の次数を直接自動微分と照合。三つの振幅で勾配分解の誤差3.47e-18以下、平均信号/s³ は解析係数0.4789125に近づく。

- `adam_cumulative_positive.json`: balanced label enumeration（iid ラベルではない）、二段 shared CNN、全 Conv/head actual Adam 200 更新。mean 下降0.07159685561、norm/variation による下降下界0.06551812723、厳密な累積恒等式との誤差7.50e-16未満。全時間の単調性は companion note の構造的証明による。


## 課題内ラベル再利用・負mean・本来の空間Convの追加検算

- `task_reuse_longtime.json`: 10 classes、4 images、batch2、8更新/task、80tasks。同じtask内のlabelを固定し、640回の全raw共同SGDと再帰が1.56e-15未満で一致。課題開始参照とのずれは全taskで明示上界内。無限時間は解析証明と独立監査に依る。
- `task_reuse_finite.json`: negative mean、混在gate、rank2、trainable biases、全9iid label assignments。実期待下降約1.41899461157e−5、解析下界9.59373196875e−6。full/self NTK容量微分は厳密な有理数で正。初期の両微分が半径.005での変化上界.275625を上回るため、球全体でも正。
- `adam_task_reuse.json`: 旧taskから全momentsを保持した通常Adam。独立な次task割当を各3回reuseし、参照期待と誤差を外向き有理区間で評価。期待下降下界1.976479684e−5以上。実Conv/autogradとの比較は補助確認。
- `full_ntk_positive.json`: 本来の5×5 spatial Conv二段、全bias、全444raw params。全座標の有限摂動後も非比例channelでfull/self容量とCE方向に正の余裕。数値値はfloat64の照合であり、非空な開集合の主張は構造式・strict margin・有限摂動定理による。追加hidden FCは解析的に拡張し、新たな大規模学習実験は行っていない。

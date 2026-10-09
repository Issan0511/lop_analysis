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

18. 通常のiid binary labelsを課題内H回再利用し、過去momentsを保持した全raw共同Adamの無限時間正例を得た。二段1×1 Conv・複数Pool・36個の自由なraw weightsについて、対称な初期状態をAdamが厳密に維持する。指定の減衰・damping付き学習率なら、任意に高く指定した事前成功確率で振幅aが0へ収束し、実学習率の総和は発散する。全状態で元のfull/self NTK容量方向が一致する。bias無し・binary・rank1特徴・full-batchの限定構成であり、通常一定学習率のRL-CIFARではない。

19. 等重みの不変線を外し、native5×5 Conv二段・hiddenFC二段・全biasを含む全raw空間の開集合で、実Adamの高確率長期net沈降を証明した。単一画像・binary iid task labels・H回reuse・power-decaying scalar learning rateが条件。正のhead contrastを残して出力biasで均等予測の近傍を作り、平均ODEの終点写像πとz²のobservable平均化により、全raw軌道の永久保持・全パラメータ収束・strict mean低下を同時に導く。amplitude-dependent dampingは不要。hiddenは正で残り、実RL-CIFARのmany-image/10-class/一定学習率までは未証明。

20. 単一画像の制限を、異なる正画像3枚・rank3の特徴とlogit Jacobianへ緩めた。native Conv5/FC/全bias、独立raw初期開集合、binary iid task labels、H回full-batch reuse、moments保持、十分小さいpower-decaying学習率での高確率長期net沈降を示す。多出力の平均ODE終点πを定量的変分評価でC2と証明し、初期ray D*uのmean lossを正の二次形式 (JDu)^T(JDJ^T)^−1(JDu) から導く。πとΣlogcosh(z_n)のobservable平均化で、実Adamの永久保持と最終mean低下へ接続する。容量の近傍保証は指定ridgeについて。任意画像・10分類・shuffled minibatch・一定学習率は未解決。

21. full-batch条件を、各epoch独立にshuffleして1枚ずつ更新するsingleton minibatchへ緩めた。taskの同じ画像labelを有限E>=2 epochs再利用し、全raw通常Adamのmomentsを継続する。task/imageごとのEMA集約でexact定常線形化Lを求め、出力biasを厳密に分離した有限スペクトル不等式でSym(JL)>0を保証。last hiddenFC/headの小さい初期scaleで非空性を構成し、非対称正常方向のC2終点とray LLᵀuから、strict最終mean低下の全raw初期開集合を得る。native構造・元の容量自己項との接続を保つ。binary・特殊な3画像・小さい非bias感度・減衰学習率は残り、batch16/10分類/一定学習率は未解決。

22. singleton条件を外し、N32またはN48のbinary画像を各epoch独立に再shuffle・再編成してbatch16で更新する全raw通常Adamへ拡張した。E2 epochs/task、標準beta/epsilon、moments保持のまま、課題単位の相関を保ったoutputbias応答の正下界と非bias相対スペクトル条件から正常安定性を導く。balanced batchで勾配0が可能な点は、全raw同時のgood-task eventとinverse RMS momentsにより期待場のC3を証明して扱う。既存の非対称終点写像とobservable平均化により高確率strict最終mean低下へ接続する。binary・特殊画像・短いtask・小さい初期scale・減衰率は残り、標準RL-CIFARの10分類/400epochs/一定学習率は未解決。

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

## iid課題反復で全raw Adamが動く長期正例

- `analysis/cnn_drive_1009/adam_iid_longtime.md`: 二段shared Conv、36rawweights、通常binary iid labels、課題内full-batch再利用、moments保持。正値はpathwiseに証明し、上側領域の永久保持を平均化誤差の高確率上界から導く。有限時刻ではReLUは正側、極限ではmeanが0になる。先のSGD構成のa∞>0とは初期条件・更新法が異なる。
- `analysis/cnn_drive_1009/adam_task_averaging.md`: moving Adam履歴と固定状態の定常task-block応答を接続。EMA履歴の追跡誤差は絶対総和有限、残る依存雑音はPoisson/martingale分解で扱う。task内の相関を捨てず、v=0境界でも成立する。各binary raw coordinateの定常符号は、独立taskごとの対称性から導く。
- `adam_iid_longtime.json`: 全36raw coordinatesを60課題×4更新し、再帰・勾配・moments・全NTK/full-self方向を照合。最大差6.67e−16未満。有限軌道を無限時間の証明や成功率推定に使わない。
- 同JSONのexact rational certificateは、a0=.2、H4、標準β・epsilon=1e−8で、事前成功確率90%以上の条件が具体的な正のdelta0≈1.0072e−16で満たされることを確認する。これは非常に保守的な存在証明用の上限で、実用的な学習率への保証ではない。有限raw検算のdelta0=.01とは明確に分ける。

この結果は、固定状態のAdam定常応答だけでなく、実際に動くCNNの無限時間へ接続した限定正例である。失敗事象を含めた無条件の期待沈降や、標準RL-CIFARの長期Adamは依然として結論していない。

## native全bias・非対称な初期状態の開集合での長期Adam

`analysis/cnn_drive_1009/adam_open_longtime.md` が統合定理。native参照、終点写像/flow tube、非線形observableの平均化をそれぞれ保存し、別々の独立監査と統合監査を通した。厳密なweight equalityを要求せず、全raw coordinateを独立に動かせる初期開集合を扱う。共有Conv・Poolの非winner・全biasを省略しない。

平均化参照の終点πは解析上の写像であり、実Adamへのprojectionではない。πとz²の累積誤差は収束し、その全時間上界から中間stepを含む非退出を導く。所定の成功確率1−alphaは、共通の定数・学習率上限を使った各初期点の保証であり、非可算な全初期点が同じnoise realizationで同時成功するとは言わない。

`adam_open_native.json` は331自由raw params・Conv5×5二段・FC二段・全biasの局所式の検算。全/自己K′の参照下界は約6.58536/4.20821、有限摂動誤差は約1.08e−4/8.96e−5、最大式差3.56e−15未満。二つのConvのchannel mapは非比例。stationary phase平均での終点mean損失係数は約2.01556で、block-sumとのH倍は比で相殺される。これは長期成功率や終点の測定ではなく、今回の開集合族への具体的な確率認証済み学習率も算出していない。非空性と長期主張は解析証明による。

出力biasによる相殺はfull networkだけに行う。literal selfのCE符号は一般に一致しないが、元の全rawNTK容量のself方向は正のままなので、実full mean低下との接続は保たれる。1画像・2分類・十分小さい減衰学習率・局所routingという制限を残す。

## 異なる3画像・rank3への長期Adam拡張

`analysis/cnn_drive_1009/adam_three_images_longtime.md` が統合定理。3画像の平均入力は元の1画像と同じに保ち、共通head contrastのlevel面に沿って異なる画像を作る。特徴行列とlogit Jacobianはrank3を持つため、同じ画像の複製や同じ出力方向への還元ではない。画像ごとのiid binary labelsを同一task内H回full-batchで再利用し、全raw parametersと過去momentsが動く。

3という奇数を使うことで全8ラベル配置で各raw勾配に正の絶対値下限があり、定常Adamの正対角係数Dが滑らかになる。正常方向の安定性はΣlogcosh(z_n)で示し、終点πのC2性は一次・二次変分の明示上界から証明した。単に安定多様体という名称だけには依存しない。初期方向D*uと有限の全raw摂動coneから、最終mean低下の正marginを導く。

`adam_three_images.json` は331raw params・native Conv5/FC/全biasの有限検算。特徴とlogit Jacobianの最小特異値は約6.27e−4と5.24e−4、全8割当の最小raw勾配は約1.82e−4。ridge=1でfull/self容量方向の有限連続性誤差後の下界は約4.77744と4.75482。CE式誤差は5.56e−17未満。任意の正対角metricでprojection恒等式も照合するが、そのmetricを実Adamの定常Dや初期coneの数値認証に流用しない。数値はfloat64の検算で、解析的非空性・長期確率証明と区別する。

元の容量自己項は指定したridgeで正のまま保たれる。3画像を同一に近づけるとrankの最小特異値が0へ近づくため、全tauや全ridgeに一様な保証とはしない。実RL-CIFARの一般画像・10分類・shuffled minibatch・一定学習率、負meanやReLU停止は依然として結論しない。

## ランダム順序のsingleton minibatchへの長期Adam拡張

`analysis/cnn_drive_1009/adam_singleton_longtime.md` が統合定理。binary labelを各画像に独立に割り当て、task内の複数epochでそのlabelを再利用し、各epochのrandom permutation順に1枚ずつ学ぶ。full-batchの定常符号補題は流用せず、同一task/imageに属するEMA重みを集約したexact線形化を使う。均等出力でsingleton gradientの二乗がlabelに依存しないことが鍵となる。

Lの要素は対応するraw感度と同符号だが、それだけではJLの安定性を結論しない。output-biasの±1/2はshuffle対称性により正の共通画像行列として厳密に扱い、残りのJacobian Rに対して 2(max|R|/epsilon)||R||op||R||F<sigma_min(R)^2 という有限条件を用いる。last hiddenFCのweights/biasとhead contrastを正のtでscaleするとR=[tJ1,t²J2]、J1はhead-feature blockからrow rank3を保つ。誤差O(t³)と正常方向の余裕O(t²)を比較して、厳密に正の有限tが存在する。

平均ODEは局所的にFbar=V(theta)zとなり、Sym(JV)>0からz²が減る。非対称行列でも一次・二次変分の定量証明を適用してC2終点πを構成できる。初期ray v=LLᵀuは正常空間に属し、初期meanから終点meanへの差の一次係数が||Lᵀu||²>0となる。有限coneとπ/z²のobservable平均化により、実Adamの全phase非退出、parameter収束、strict最終mean低下を導く。個々の更新の同符号は仮定しない。

`adam_singleton.json` は331raw params、3画像、2epochs/task、H6、epsilon=1e−8の式検算。参照の層scaleは1e−18で、実用的初期化や学習率の提案ではない。安定条件の誤差比は約5.38e−5、正常方向の正下界は約4.33e−35。容量微分/t²はfull約2.74765、self約2.01236。2tasks×3labelsの全64履歴を列挙した12-step有限EMAの線形化は、座標相対誤差5.50e−16未満で一致する。これを定常係数や長期成功率の実測とは扱わない。出力biasだけは無限定常履歴のphase共分散を有理数で厳密に計算し、正の共通画像係数約1.99630を独立な計算法でも照合した。

元のfull/self容量はK=Kbias+t²K1+t⁴K2を使い、容量微分/t²の画像幅とtの共同連続性から同じ有限参照で正と示す。t=0そのもののReLU微分を使わず、正側の多項式係数の連続延長である。容量の局所方向一致であり、全パラメータ更新中の容量値そのものの単調下降は結論しない。

普通のepoch shuffleと課題内reuseは扱えるようになったが、batchサイズは1。binary・特別な近接3画像・小さい非output-bias感度・局所routing・十分小さい減衰学習率という条件を保持する。標準RL-CIFARのbatch16/10分類/一定学習率、無条件期待沈降、負meanと死は依然として未証明。

## batch16・毎epochの再編成への長期Adam拡張

上のsingleton段階に続き、`analysis/cnn_drive_1009/adam_b16_longtime.md` でbatch16を扱った。N32またはN48の異なる画像に課題ごとに独立binary labelsを割り当て、E2 epochsの各々で新しく全画像をshuffleしてbatchを組み直す。同一課題内のlabelを再利用し、Adamのmoments・global bias correctionは継続する。標準beta=.9/.999、epsilon=1e−8を変更しない。全raw独立な初期開集合、十分小さいpower-decaying学習率、元のfull/self容量方向との接続を維持する。

16枚の平均ではlabelの打消しがあるため、singleton時の決定論的な勾配floorは使えない。代わりに「あるbatchのラベル数が8対8でない」という全raw共通の事象から、compact全体に一様な定常RMSの逆モーメントを得る。Hilbert normの微分を使うとC3期待場に必要なのはinverse-second momentであり、証明書ではinverse-eighth momentまで余裕を確認した。有限履歴の個々のAdam写像がRMS0でC3だとは主張しない。

outputbiasは感度±1/2で小さくならないので別証明にした。task間独立性とtask内相関を保持したEMAのsecond/fourth moments、RMSの下側確率評価から、共通contrast方向の定常応答Gammaの全phase下界をH4で3.3279993247、H6で2.9847166156と得た。`adam_b16_certificates.json` の有理数証明書でいずれも厳密に2.9を超える。これをNで割りH phase分を足した正の共通画像行列がoutputbiasの正常方向寄与となる。このbiasの正下界はE2に限り、逆モーメントの任意有限Eへの拡張と混同しない。

残るraw感度Rには C_B=1+sqrt(N/16) を使った相対応答誤差があり、C_B(max|R|/epsilon)||R||op||R||F<sigma_min(R)^2 が正常安定性の有限十分条件になる。native正画像の同head-level simplexとlast hiddenFC/headのscaleでrankNとこの条件を両立する。特異値の小さい画像差モードも含む条件であり、画像が一致する極限に一様な余裕は主張しない。正ray LLᵀu、C2終点、pi/z²のobservable平均化から、実軌道の全phase保持・全parameter収束・strict mean差を導く。

`adam_b16_native.json` の有限照合はRGB24、Conv5(2)/Conv5(2)、FC32/32、全3712raw、N32、E2、B16。画像幅.002、参照scale約5.4814e−19、相対感度変化.0007214未満、スペクトル誤差比約.1、正常方向下界約5.1314e−38。容量微分/t²はfull約63.43024、literal self約42.21268で正。4種類の固定labelと各々2回の独立shuffleで得た16batchesのCE恒等式は、感度で割った最大誤差1.05e−15未満だった。8対8のbalanced batchも含む。これらはfloat64の式照合で、全2^32label列挙、定常応答実測、長期学習、成功率推定、具体的な確率認証済み学習率の算出ではない。

ReLUは正側で動き続ける。今回外せたのはbatchサイズ1の条件であり、binary・特殊な近接画像・短いtask・極めて小さい有限初期scale・局所routing・十分小さい減衰学習率は残る。標準RL-CIFARの一般画像・10分類・400epochs/task・一定学習率、負mean、gate停止、無条件期待沈降は未証明。

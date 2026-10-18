# neokunekune_cifar5p1_cnn_1018 — 実ラベル5+1 CIFARでの損失ゲートSnake

状態：**実行前spec v1・実装／学習未実施**。2026-10-18、Codex。
依頼：Issa「まずは実践投入かな」→5+1 CIFARのCNNを第一候補として提案→「spec書いて」。
本specの保存・登録と、GPU実験の起動は別段階とする。

## 1．目的と今回検証する範囲

乱数ラベルCIFARのCNNで有効だった、学習損失に応じて分散EMAの更新を遅くするSnakeを、**実ラベル・入力分布変化のある5+1 CIFARのCNNへ移植する**。仮称neokunekune、実験腕名は `NEO` とする。関数形は通常のSnakeのまま。

問うことは、(a)通常の幅適合より再学習が速いか、(b)課題1で尺度を合わせて固定する方法に対して性能を保てるか、の二つ。ReLUを基準として同時に測定する。幅の持続成長やReLUの崩壊を、本走を実施するための通過条件にはしない。

主指標は現在課題の**更新前online精度**であり、課題末test精度は別に報告する。過去クラスの保持・100クラスの累積認識・長期の尺度安定性・最適に調整した既存手法への優越は、この実験だけでは主張しない。

既知の設計根拠は `specs/spec_sna_loss_gate_1018.md` と `specs/spec_sna_loss_gate_1018_production.md`、その結果 `results/sna_loss_gate_1018/cnn/summary.md`。5+1のデータ・課題構成は `src/cifar5p1_mlp_0920.py` を基準とする。これらは既に観測した開発根拠で、本実験を発想から独立した予言とは扱わない。

## 2．固定する実験条件

| 項目 | 固定値 |
|---|---|
| データ | CIFAR-100 fine labels、train 50,000枚・test 10,000枚 |
| 課題列 | 30課題、1-based奇数がhard、偶数がeasy。hardは5クラス×500枚、easyは1クラス×500枚 |
| クラス消費 | seedごとに100クラスを並べ替え、先頭90クラスを5,1,5,1,…で消費。再訪なし |
| ラベル・出力 | 元の0–99ラベル、**出力100次元**。課題内ラベルへの付け直し・現在クラスへの出力maskは行わない |
| 入力 | float32のu8/255、CIFAR-100固定チャネル標準化。mean=(0.5071,0.4865,0.4409)、std=(0.2673,0.2564,0.2762) |
| augmentation | なし |
| CNN | Conv5×5 3→16 pad2 → φ → MaxPool2 → Conv5×5 16→16 pad2 → φ → MaxPool2 → Flatten1024 → FC100 → φ → FC100 → φ → Linear100 |
| bias | 全5層にあり |
| 正規化・正則化 | BN/LN・中心化・weight decay・重みcap・bias制約・dropoutなし |
| 初期化 | 各層のW・bをU(−1/√fan_in,+1/√fan_in)。CPUの独立init stream、層順にW→b。4腕で同じ全パラメータ |
| optimizer | Adam、lr=1e−4、β1=.9、β2=.999、ε=1e−8、bias correctionあり。gradient clipなし |
| 学習予算 | **hard/easyとも780更新/課題、batch32**。30課題で23,400更新/run |
| 課題間 | 全W・bias・Adam m/v/時刻・活性化状態を継続。headをリセットしない |
| 本走seed | **2000–2009の10seed**。各seedを4腕に対応させる |
| 計算 | float32、AMPなし。matmul/cuDNN TF32ともfalse、cuDNN benchmark=false。決定的演算を要求し、対応しない演算は黙って切り替えず記録して実装段階で解決 |

ネットは `src/rlcifar_cnn_0908.py` の4活性化サイトc1/c2/f1/f2を引き継ぎ、出力とbatchをこの箱に合わせる。既存RC runnerの1200画像・10出力・batch16という大域定数をmonkeypatchしない。新adapterを `src/neokunekune_cifar5p1_cnn_1018.py` に置く予定とする。

lrは5+1の既存診断起点で、各手法の最良値という意味ではない。今回の固定設定で得た結果を、lr・形状を調整した全手法の順位へ広げない。

## 3．4腕と尺度の更新則

| 腕名 | 活性化 | 課題1 | 課題2以降 |
|---|---|---|---|
| `R` | ReLU | 通常学習 | 通常学習 |
| `SNA` | 通常の幅適合Snake | 通常適合 | 通常適合を継続 |
| `SNAfrz1` | 課題1末固定Snake | 通常適合 | 課題1の最後の更新後のαをsite・unit/channelごとに固定 |
| `NEO` | 損失ゲートSnake | 通常適合 | 学習CEに応じて分散EMAの更新速度を制御 |

Snakeの全3腕は、課題1末の校正・固定を行う直前まで同一の手順で学習する。

\[
\phi(z;\alpha)=z+\frac{\sin^2(\alpha z)}{\alpha},\qquad
\partial_z\phi=1+\sin(2\alpha z).
\]

各siteの分散状態はV=1から開始する。

\[
\alpha=\operatorname{clip}\left(\frac{0.6}{\sqrt V},0.005,3\right).
\]

convでは出力channelごとにbatch×全空間位置上の母分散（unbiased=false）、FCではunitごとにbatch上の母分散を使う。convのαは空間位置で共有する。V・α・ゲートはdetachし、学習可能パラメータへ加えない。中間の丸めを避けるため、関数計算の演算順も既存Snakeの `z + sin(alpha*z)**2 * alpha.reciprocal()` を用いる。

### 3.1 課題1の基準損失

課題1はhard。更新前のミニバッチ平均CEをL₁,…,L₇₈₀として保存し、**最後78更新（j=703,…,780）**の中央値から、

\[
L_{\rm ref}^{\rm raw}=\operatorname{median}(L_{703},\ldots,L_{780}),\qquad
L_{\rm ref}=\max(L_{\rm ref}^{\rm raw},10^{-4})
\]

を計算して永久固定する。偶数個の中央値はソート後の中央2値の平均。全3Snake腕で診断用に計算するが、制御に使うのはNEOのみ。seed・腕をまたいで共有しない。

**RC版からの事前指定の変更はこの集約窓である。** RC版は400epoch中の最後10epochのepoch平均CE中央値だった。今回のhardは約10epochしかなく、同じ「最後10epoch」では初期の学習過渡までほぼ全て混ざるため、最後10%の更新に定める。式・c・β・clip・損失床はRC版から移植する。これは完全に同じ校正手順の再現とは書かない。

実ラベルでLrefが大きくても、NEOのgateがほぼ1のままでも、結果を見て窓・床・倍率・参照値を変更しない。新規則を試す場合は別spec・別run idにする。

### 3.2 課題2以降

NEOの各更新で、現在のforwardから得た**更新前・100出力に対する平均学習CE**をLとし、

\[
g=\min\left(1,\frac{L_{\rm ref}}{\max(L,10^{-4})}\right),\qquad
\beta_{\rm eff}=0.01g,
\]
\[
V_{j+1}=(1-\beta_{\rm eff})V_j+\beta_{\rm eff}\operatorname{Var}_{\rm batch}(z_j)
\]

とする。同じrunの4siteへ同じgを適用する。SNAは常にβ=.01。SNAfrz1は課題2以降のVも固定し、診断用の実測分散は別変数へ記録する。

1更新の順序は、(1)更新前Vからαを作る→(2)forward・CE・online当たり率→(3)detachしたCEからgを計算→(4)当該αを固定した逆伝播とAdam更新→(5)同じforwardで得た更新前zの分散でVを更新、に固定する。新しいVは**次のforward**から使う。新CEやtest lossを得るための追加forwardは制御へ使わない。

課題2以降の参照値再校正、easy/hard別の参照値、損失のクラス数補正は導入しない。easyでも100出力のCEを使う。初期校正で課題1末という境界を使うため、境界情報不要の手法を実証したとは書かない。

## 4．対応づけ、データ列、fresh対照

課題列は既存 `task_plan(seed)` を再利用する。batch列は、各task内の全画像の独立randpermを必要数だけ連結し、先頭24,960枚を(780,32)へ切る。末尾の不完全epochを捨てたり、epoch境界でbatchを小さくしたりしない。

初期化・クラス順・batch列・診断画像抽出の乱数streamを分ける。クラス順は既存 `c51_classes` streamと同じ。batchはseedごとに新しい `c51_batch` streamを作り、task1から30まで既存順に消費して**全課題分を訓練前に確定・保存**する。診断や腕数の増減が列を変えないようにする。全パラメータ・課題列・各taskの画像ID／batch列のSHA256を保存する。

新seedは既存の主比較seed0–9やRCの10–19と分けるために選んだもので、同じCIFAR-100データを使う。未使用データによる外部検証とは呼ばない。

**task29の同課題freshを全腕について報告する。** 初期W・b、Adamゼロ、時刻0、V=1、未校正・未固定の制御状態へ戻し、本走のtask29と全く同じ780batch列を学習する。fresh側では当該課題が最初の課題なので、3Snake腕は全期間通常SNAとなる。校正・固定は当該課題の終了後にしか起きない。

3Snakeのfreshが同一であることを検査できれば、計算は1本に共用してよい。共用した事実と対応hashを保存する。継続側のLrefやαをfreshへ移さない。fresh gapはfresh−継続と定義し、腕間でgapが小さいだけで高性能と判定せず、両方の精度を併記する。

## 5．評価量と登録判定

### 5.1 主量

各seed・各腕について、各hard課題の780更新の**更新前当たり率**を平均する。seed単位の主量Aは、後期hard **t=21,23,25,27,29** の5課題平均。easyは主量へ混ぜない。

比較は全て同じseedの対応差を先に取り、10seedで平均する。順位表の別々の中央値から優劣を決めない。数値は内部では0–1、表では%とポイントを使い分ける。

| 問い | 対応差 | 基準 |
|---|---|---|
| 通常適合より改善するか | D₁=A(NEO)−A(SNA) | 区間下端>0で `BETTER_THAN_SNA` |
| 課題1末固定に対して性能を保つか | D₂=A(NEO)−A(SNAfrz1) | 区間下端>−.005で `NONINFERIOR_TO_FREEZE` |

2比較について、seedを再標本化するpercentile bootstrapを**20,000回、NumPy default_rng(1018)**で行う。同じ再標本化indexを両差へ用いる。各比較の両側**97.5%区間（分位点.0125,.9875）**を判定に使い、2比較にBonferroni配分した近似的な同時95%区間として扱う。bootstrapは有限標本での被覆率を保証しない。10seedは既存RCと同規模の確認で、検出力計算に基づく保証はない。

非劣性幅.005＝0.5ポイントはRCで使った許容差を持ち込んだ実用上の選択。誤差や有意差を見て拡げない。区間が境界に等しければ未達。D₂の下端>0なら固定に対する優越も同じ区間で報告できるが、非劣性だけで「同等」「固定より優れる」とは書かない。

両判定を満たした場合に「この固定設定の実ラベルCNNで、通常適合を改善し、課題1末固定への登録幅内の非劣性を満たした」と記述する。片方だけ満たす場合はそのまま分けて報告する。区間が閾値をまたいだ場合は差なし／同等と結論せず、`INCONCLUSIVE` と対応差・区間を示す。

### 5.2 補助量

- 全15hardのonline平均と全task曲線。早期窓t=1,3,5,7,9・後期窓は上記とし、早期−後期も示す。
- hardごとの最初100更新のonline精度。全780更新の主量と区別する。
- 各課題末のtrain当てはまりと**現在課題クラスのtest画像**の精度。予測は100出力argmaxのままで、現在クラスへmaskしない。全hardのtest平均と後期5hard平均を別に報告する。
- task29のfresh online・test・fresh gap。
- NEO−R、SNA−R、SNAfrz1−Rの後期対応差・通常95%bootstrap区間。補助的比較であり、上の2比較と同じ確認的優越判定を増やさない。
- seed別の値と対応差を全件表示。平均と区間だけで個体差を隠さない。

online改善とtest改善は別の結論にする。ReLUの早期−後期やfresh gapは設定の特徴として報告するが、本走の成功をReLUの死亡率へ依存させない。

## 6．制御と幅・応答の記録

**10更新ごとのblock記録**（task内78block）：更新前CE平均、gの平均・最小・最大、Σβeff、gate有効フラグ。各taskの最初10更新・最後10更新はCEとgのstep値も残す。Lref_raw・Lref・下限発動フラグ、task1の校正に使った78個のCEを保存する。

各siteでV・α・clip下限／上限の該当数とunit/channel数、task開始から終了までのαの変化を保存する。gが動いていてもαがclipで変わらない場合を区別する。NEOが通常適合とほぼ同じ、ほぼ固定、序盤抑制・終盤再開のいずれに近いかは記述し、都合のよい軌道だけを選ばない。

**診断入力は二種類**：現在課題の全train画像と、全100クラスから20枚ずつ選んだ固定probe 2,000枚。probeはCPU独立stream（固定seed=1018、役割名`neo_c51_probe`）で選び、画像IDを保存する。通常trainに含まれる画像を使う幾何診断で、独立testとは呼ばない。probeの未来クラス画像を勾配更新や校正へ使わない。

課題末にc1/c2/f1/f2を別々に測り、unit/channel別m・s・r=m/s、導関数の平均・SD、平均|導関数|を保存する。s=0はr未定義としてmaskし、その件数を記録する。sは母SD、統計の集約はfloat64で行い、推論はfloat32のまま。層集約はunit/channelの中央値・四分位とし、convの16channelの中央値は中央2値の平均に固定する。

convのsは、(i)画像×位置の全体SD、(ii)位置ごとの画像間分散の空間平均の平方根、(iii)位置平均の空間SD、を分ける。(i)は適応則の分散粒度に対応するが、位置間の定常的差も含む。(ii)を画像への応答の幅として併記する。中央1位置の結果だけでconv全体を代表しない。

さらに実測sに対する2αsを保存し、内部EMAの√Vに対する2α√Vと区別する。重み行ノルムは補助に記録するが、sの代用にしない。診断ではV・α・Lref・Adam・乱数状態を更新しない。probeはchunkに分け、必要な統計だけを集約する。

幅の増大自体は失敗としない。性能・制御・実測幅を併記し、精度改善だけから「過渡幅の追随が原因」「幅の暴走を解決」「機構を同定」とは書かない。

## 7．実装後・本走前の検査と不成立時の扱い

本specの作成では学習を実行しない。実装する際は次を確認し、検査結果とコードcommitを本走開始前に固定する。

1. データshape・クラス枚数・SHA256、課題列のhard/easy・90クラス非再訪・100出力、batch列の所属を検査する。
2. 全4腕の初期W・b・全課題batch hashが一致し、3Snake腕の課題1の更新・CE・V・αが一致する。各runのCEから独立してgateが決まることも検査する。
3. 保存したCEからLrefをCPUで再計算し、最後78個・偶数中央値・floorに一致する。高損失／低損失の人工入力でgate式、βeff、更新順を検査する。
4. SNAfrz1がtask2以降にα・Vを変えず、NEOも診断forwardでは状態を変えないことを検査する。
5. 同じ状態からの単独run／束ねrun、eager／CUDA graph、保存再開／連続実行を比較する。異なる演算経路にbit一致を無条件に要求せず、同経路の再開は一致、異経路はfloat64参照と演算から定めた誤差・データずらし等の変異で判別可能な検査にする。実行経路と演算精度を全腕で揃える。
6. freshは初期全状態へ戻り、継続task29のbatch hashを共有する。3Snakeのfreshを共用する場合は、各腕名での最初の課題の動作が一致することを確認する。

実データsmokeはseed9000・9001、最初2課題のみ、全4腕で行う予定。full-780更新の動作、校正境界、固定、再開、計時を検査するためで、主比較seedへ混ぜない。smokeの性能でlr・ゲート規則を選ばない。規則を変えた場合は変更理由・観測範囲を残し、新版を登録してから本走を開始する。

非有限のCE・勾配・重み・Vを検出したrunは、失敗直前状態とtask/stepを保存して停止する。黙ってlrを下げたり、seedを交換したりしない。装置障害は同一保存状態から再開し、理由をprovenanceへ追記する。アルゴリズムの失敗や未完走seedが一つでもある場合、完全な10seedを前提とする上記の確認的判定は `INCOMPLETE/FAILED` とする。完走seedだけの参考集計は分母と失敗率を併記し、成功判定に置き換えない。

seed2000–2009は全て完走まで同じ予算で実施する予定とし、途中の成績による停止・seed追加をしない。これを越える長予算・別lr・他baseline・CIFAR-100二値50課題は、本specの自動追加項目にしない。

## 8．計算量、保存、来歴

- 継続学習：4腕×10seed×30課題×780＝**936,000更新**。
- fresh：2種類（R／通常Snake）×10seed×780＝**15,600更新**。Snakeの3腕を共用する場合の数。
- 合計：**951,600更新**、batch32で30,451,200画像提示。診断forward・smokeは別。freshを4腕別々に実行する場合は967,200更新となり、実数を記録する。
- wall timeは未計測。既存MLPの時間をCNNへ外挿せず、smokeで学習・診断・保存を分けて計時する。

予定出力先：`results/neokunekune_cifar5p1_cnn_1018/`。

| ファイル | 内容 |
|---|---|
| `protocol.json` / `provenance.json` | spec SHA256・登録commit・sourceと依存hash・開始時のclean git状態・実行設定・GPU/PyTorch/CUDA/cuDNN・TF32・決定性・再開履歴・実行時間 |
| `task_plan.json` / `batch_hashes.json` / `probe_ids.json` | seed別課題と画像列・対応確認 |
| `per_task.csv` | arm,seed,task,hard/easy,online_acc,first100_online_acc,train_acc,test_acc,開始時CE等。主訓練で1,200行 |
| `per_block.csv` / `calibration.json` | 制御の時系列・step抜粋・基準損失と元データ |
| `site_stats.csv` / `unit_stats.npz` | 現在課題／固定probeの幾何・応答・V・α。母集団と統計の定義を列名で識別 |
| `fresh_control.csv` | task29のfresh・継続・gap、共用した計算とhash |
| `checks.json` | 上記検査とそのコードcommit、smokeの格 |
| `summary.md` / `verdict.json` | 全seed値・対応差・区間・2判定・失敗有無・主量と補助量の区別 |
| `state.pt` / `checkpoints/` | 再開用状態。初期、task1・15・28・29・30末のW/b・Adam・活性化状態を保存 |
| `backup_manifest.json` | 生データの退避元・退避先・bytes・SHA256 |

本specは実験前にcommit・pushする。実装・検査のcommitも本走開始前にpushし、そのhashを保存する。結果が出た後に登録文を遡って書き換えず、追補・結果ノートに成否を記録する。実行完了後は生データを `/home/issan/Projects/obsidian-research-data/neokunekune_cifar5p1_cnn_1018/` へ照合退避し、manifestと結果をmainへ統合して専用worktreeを片付ける。

## 9．参照

- 既存のRC手法：`src/sna_loss_gate_1018.py`、`src/sna_cnn_cause_1009.py`。
- CNN形状・初期化・conv尺度：`src/rlcifar_cnn_0908.py`。出力/batchは本specの値を使う。
- CIFAR-100 reader・標準化・課題列・batch生成：`src/cifar5p1_mlp_0920.py`。MLPの結果をCNNの結果として流用しない。
- [Lillo・Cheney, Activation Function Design Sustains Plasticity in Continual Learning, v4](https://arxiv.org/html/2509.22562v4)、§5・App.E。benchmarkへの接続に使い、ネット・入力前処理・初期化まで完全再現したとは扱わない。
- Obsidian結果ノート：[[損失で尺度適応を遅くするSnake_CNN結果_1018]]。

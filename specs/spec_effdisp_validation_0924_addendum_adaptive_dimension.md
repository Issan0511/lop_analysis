# 0924 追補3 — 適応Snake、Adamを主比較、SCRの入力次元

本走前、Issaの追加依頼による変更。先のspec290c4f8、追補2ed7321・89b8789は履歴として保持。未実行の腕を以下へ置換/追加する。

## 主比較の最適化器

RL-MNIST/PMの両方をAdam lr=.001, beta=(.9,.999), epsilon1e-8で揃える。RL400epoch×75step、PM1epoch×625step、50tasks、seed100–104は維持。
PMで従来登録したSGD6活性化は実行しない。代わりにSGD lr=.01のleakyだけを参照として残す。PM Adamは既存canonical SGDの明示的optimizer変更であり、厳密再現とは呼ばない。
CondAは元からAdamが主比較で、SGD1腕は参照として維持。

## 適応Snake

phi(z)=z+sin²(αz)/α、微分1+sin(2αz)。αは学習勾配へ含めず、unitごとにα=clip(c/sqrt(Vema),.05,3)。Vema初期値1、EMA更新Vema←.99Vema+.01Var(z)。
c=.3/.6/1の3腕を追加。MNISTは各隠れ層、pre-update minibatch前活性のpopulation variance（分母batch）を使い、optimizer更新後にEMA更新。既存H.AdaptiveSnakeと同じ順。
CondAはbatch1なのでミニバッチ分散が0に退化する。代わりに既知の条件付き入力分布での正確なVar(z_i)=.25 sum_random w_ij²をpre-update重みから計算し、更新後にEMAへ入れる。この推定方法の環境間差を明記する。
Vema/αは全task末に保存。勾配照合はαを固定し、EMAの順序はgraph/eagerで照合。

MNIST主比較はLR,R,ELU,GELU,SiLU,SN1,SNA03,SNA06,SNA1の9活性化。CondA m20/r5/k1にも適応3腕を追加。

## SCR 20→40bit

追加5腕、全て100tasks×T10000、seed100–104、Adam .001、epsilon1e-8、learner hidden100、LTU teacher hidden100/beta.7。

| m | random r | persistent f | boundary flips k | activation |
|---:|---:|---:|---:|---|
| 40 | 5 | 35 | 1 | leaky .1 |
| 40 | 10 | 30 | 1 | leaky .1 |
| 40 | 10 | 30 | 2 | leaky .1 |
| 40 | 10 | 30 | 1 | adaptive Snake c=.6 |
| 40 | 10 | 30 | 2 | adaptive Snake c=.6 |

CondAは総計22腕。m40/r5はrandom rankを保ってpersistent成分だけ増やす比較。m40/r10はbit構成比を保ち、k2は反転するpersistent成分の割合も元の1/15に揃える。
入力次元変更で教師関数と初期化の次元も変わる。教師幅を固定しても課題難易度・出力分散が同一とは仮定しない。単純な損失の大小を次元だけの害と呼ばない。共分散はdiag(0×f,.25×r)、supportは最大1024点。

## 追加予測（Codex）

1. 適応Snakeではα×前活性幅が追随し、負側の微分消失によるLOW_RESPONSEは固定ReLU/ELUより少ないと予測。位相と平均の効果は残るので機能的学習の保証はしない。
2. m40/r5ではrawノルムへ寄与するが条件付き幅には見えない方向が増える。raw/effectiveの乖離が広がる可能性を予測する。persistent方向は平均に効くため「無害な死重」とは断定しない。
3. m40/r10では実効変位の供給が増える可能性があるが、cも変わるので幅の単純な2倍則は予測しない。
4. PMをAdamに替えると、既存SGD参照より実効変位が大きくなる方向を予測するが、lrも異なるためoptimizer固有効果の厳密分離とは呼ばない。

P1–P6の一次基準を変えない。適応の状態量や次元比較はreport-onlyとして値とseed差を全掲載。所要時間見積りはtask末I/Oをepochごとに重複計上せず、訓練chunkの実測から出す。全設定をローカル実行し、確認は10分間隔。

# 実効変位の入力分布条件を分ける再検証 — 0924

Codex。2026-09-24。本走前に固定する。前回effdisp_validation_0924の全結果は既知。
Issaの指摘: SCR/Permutedでは実効的な入力も変わる。固定入力での平衡と、入力変化で収支が動く場合を分けて再検証する。
新規seed200–204の5本を全て残す。旧判定・旧データは書き換えない。ローカルGPUのみ。

## 1. 問いと適用域

固定入力でも平衡が出るか、入力分布の変化を落とすことでどれだけ見誤るか、入力固定だけで十分かを分ける。
恒等式の一致を科学的支持に数えない。入力固定はDや削り角の定常性を保証しない。
SCRの中心化Σは一定だが平均・支持点が変わる。PMではΣも変わる。この差を明記する。

## 2. CondAの2×2、k1/k7、計8腕

m20,r5,f15,LTU teacher hidden100 beta.7,learner20→100→1,leaky.1,Adam lr.001 betas(.9,.999) eps1e-8、weight decay/resetなし。
400tasks×10000steps,batch1。元runnerの初期化とRNG方式を保ち、seed・教師・初期値・random bits・flip prioritiesを同じにする。
課題tの元入力をx_raw=(persistent_t,random)、初期固定ビットで置き換えた入力をx_fixed=(persistent_initial,random)とする。

| cell | learner入力 | 教師を評価する入力 | 条件付き意味 |
|---|---|---|---|
| X1Y1 | x_raw | x_raw | 元のSCR。入力と32点上の標的が変わる |
| X0Y1 | x_fixed | x_raw | 入力分布は固定、標的は切り替わる |
| X1Y0 | x_raw | x_fixed | 入力は切り替わるが潜在32点の標的は固定 |
| X0Y0 | x_fixed | x_fixed | 入力も標的も固定する参照 |

各cellをk1,k7で実行。X0Y0の2本は同じであることが実装上の対照となる。
初回taskは4cellでbit単位同一。反転はtask末snapshot後、次taskの前。Yは元teacherが出す連続値、教師側の条件を隠れて変えない。
X0Y1はpersistent task cueを消すため、再訪するcontextに対する条件付き学習とtask ambiguityも変える。**raw/aligned差を入力平均だけの純粋な因果効果とは言わない。** factorial全体を報告する。
全32点のstart/end MSE、学習器入力のµと元contextのµ、初期µ、teacherの標的32個、全weights/bias、optimizer state、微分・active unitsを保存。前回と同じ中心化Σ=diag(0×15,.25×5)。

## 3. MNISTの2×2、計4腕

固定1200画像bank/seed。784→100→100→10、leaky.1、Adam.001 betas(.9,.999) eps1e-8、batch16、400epochs×75=30000updates/task。
input permutationを固定/毎task変更 × random labelsを固定/毎task変更。固定labelsも初回のランダムlabelsであり、真の数字ラベルではない。全4cellの初回input/labels/初期weightsを揃える。
X0Y0:固定入力固定ラベル、X0Y1:固定入力random再ラベル、X1Y0:毎task permutation・初回random labels固定、X1Y1:permutationと再ラベルの両方。
潜在画像の行IDは固定、全cell同seed同taskでbatch順を共有する。permutation/label/batch RNGは独立、固定cellでも相手側の乱数の消費を変えない。
このfactorialは統制用のPermuted/Random-label MNISTであり、canonical PMの10000枚/1epoch/真ラベルを再現するものではない。

**一次対比は全4cellの50tasks。X0Y1 leakyだけ150tasksまで継続して長期の別判定を行う。** 50task判定を150task判定で置き換えない。
全task末weights/bias/moments、per-task start/end/online accuracy、微分・active units・固定16画像のstart/end生勾配normを保存。
Σbase/µbaseと画像indicesをseedごとに保存し、各task permutation/labelsから実入力Σt/µtを正確に再構成する。巨大な重複共分散ファイルは不要。

## 4. Snakeの限定対照、追加2腕

MNIST X0Y1と同じfresh seeds、50tasks/30000updatesで固定Snake α=.05 と適応Snake c=.3を比較。Snakeは両隠れ層。
phi(z)=z+sin²(αz)/α、dphi=1+sin(2αz)。適応は前回のEMA(.01)、Vinit1、α=clip(.3/sqrt(Vema),.05,3)、pre-update minibatch population varianceをoptimizer更新後にEMAへ入れる。
LRとのデータ/ラベル/batch/初期weightsは共通。活性化が違うため初期関数は一致しない。
α、Vema、下限/上限clip率を保存。固定α=.05で再現しても、適応系の初期過渡が無関係だとまでは言わない。

## 5. 計量：学習更新と入力切替を混ぜない

各行tはWprev→Wnext。D=Wnext−Wprev、db=bnext−bprev。
中心化variance V=trWΣWᵀ。固定reference（最初のbank）を全条件で保存し、元と同じP1/P2/P3を適用。
SCRはactual/referenceのΣが同じ。MNIST X1ではactual Σtも計算する。

actualの分解を学習側の現在入力Σtで統一する:

    Vprev_old=tr(Wprev Σprev Wprevᵀ)
    Vprev_current=tr(Wprev Σt Wprevᵀ)
    Q=tr(D Σt Dᵀ), X=tr(Wprev Σt Dᵀ)
    G_pre=Vprev_current−Vprev_old
    Vnext_current−Vprev_old=Q+2X+G_pre.

元解析の『旧Σで更新を測り、新WでGを測る』分解も正しいが、今回は同じ値を混用しない。Q/X/Gの定義を列名・metadataに残す。
実際の前活性変化の平均を落とさないため、同じ潜在support/画像IDで

    A = D x_current + db                 (学習による変化)
    E = Wprev (x_current−x_previous)      (入力切替による変化)
    total = A + E.

E||total||²=E||A||²+E||E||²+2E<A,E>を分ける。A自身もQ+||Dµ_current+db||²に分ける。
CondAは全32支持点、MNISTは固定1200画像bankを用いる。weight更新0でも入力切替による変化は残り得る。
共分散/paired bankのidentity照合は帳簿検査だけ。元データの意味を持つmean channelを死重と呼ばない。

## 6. 判定と窓

前回metrics.summarize_task_tableのP1/P2/P3/P4を固定referenceにそのまま適用。ゼロQ/NaN/LOW_RESPONSEの扱いも維持。
50taskのlate41–50、150taskは121–150、CondA400taskは321–400。CondAの最初100task(81–100)もsecondaryとして全腕掲載。長期は短期に上書きしない。
群PASSは4/5seed以上。seedごとの値、seed中央値、bootstrap95%(5000draws,seed924)を保存。paired差/比もseed内で作る。
LOW_RESPONSE: いずれかの隠れ層の平均絶対微分<1e-8またはactiveunitfrac<.1が3task連続した最初のtask。停止前20遷移が無ければ不足と記す。STOPPEDは低相対更新量であって学習不能とは限らない。

### 動く実入力の補助判定（新規、一次Pを変更しない）

A_BALANCE: B_all=−sum(2X+G_pre)/sumQ ∈[.9,1.1]、sumQ/mean(Vprev_old)≥.1。
A_PLATEAU: A_BALANCEに加えてactual Vのlate logV/logtask傾き絶対値≤.2、late後半/前半の平均V比∈[.9,1.1]。
入力変化による相殺でも通るため、task更新による削りと呼ばない。update-only Bとc<0割合も併記。
G比、mean energy比、paired function-change分解はreport-only。Gを観測してから予測へ入れた値をheldout predictionとは呼ばない。

### 入力固定だけで十分か

MNIST X0Y1 leakyの50と150task、CondA X0Y1の400taskについて、P2/P4の通過/不通過を明示。
固定入力でも不通過なら『観測した地平・optimizer・活性化で入力固定だけでは十分でなかった』まで。無限時間の不存在は言わない。
入力drift効果はYを揃えたX1−X0のpaired差を全掲載し、cue/表現変化の限定を付す。effectが出た指標だけ拾わない。

## 7. 実行前予測（Codex、Issaの予測ではない）

1. MNISTのX1では入力切替のG/paired-input項が無視できず、referenceの判定とactualの判定が一致しない場合があると予測する。ただし恒等式が閉じること自体には予測価値を置かない。
2. SCRの入力固定で平均切替の項は厳密に0になる。これは対照の定義であり科学的な当たりに数えない。X0Y1では変化する標的への更新は残り、入力固定だけでP2が必ず通るとは予測しない。
3. RL leakyは150taskで50taskよりVのlog傾きが小さくなる方向を予測するが、P2の群PASSは不確実（主観50%）。
4. 固定Snake α=.05は適応c=.3に近いlateの平衡を示す可能性が高い（主観65%）。両方P2を通るかと、late R/Dのpaired比を報告。一致しなければ初期過渡の差を含む不一致とする。
5. 固定入力/固定ラベルは課題開始と終了の性能差が小さくなる。小さなDや幅停止を可塑性喪失と呼ばない。
6. 入力変化の補正で前回の不成立が全て解消するとは予測しない。収支式、有限地平の平衡、独立予測の支持は別に維持する。

## 8. 実装・監査・資源

実装はgpt-6-sol、rootが設計・数式・報告を監査。所有moduleを分ける。訓練前にspecをcommit/push、コードを固定してsource hashを各metadataへ記録。
新しい変換の同初回task、bit/label/perm/batch pairing、teacher target分離、graph/eager、固定Snake gradientのautograd照合、分解の時点と混合schemaを必要最小限で検査。
速度smokeは2task以内、性能で条件を選び直さない。本走の想定約2–4時間、既存ローカルGPUで順次実行。省略/短縮を結果に合わせて決めない。
全データをobsidian-research-data/effdisp_inputscope_0924へ直接保存。進捗確認は10分間隔。非有限値は記録して該当seedを停止、勝手にlr変更しない。
完了後はSHA256 manifest、正本results、Obsidian予測/結果、main merge/push、worktree/branch cleanupまで行う。

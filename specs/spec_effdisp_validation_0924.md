# 実効変位と幅の環境横断検証 — 0924

起草: Codex（Issa の予測ではない）。2026-09-24 JST、結果を見る前に固定。
対象: Obsidian「実効変位とWの関係_主張候補_0923」§9 の1箱限定と、V10 の上流 H1。
元ノートと公開済み RL-CIFAR の値は既知。新規 CondA/MNIST は prospective、既存 CIFAR の再集計は report/reproduction と区別する。

## 1. 検証すること

1. 非零の更新を続けながら、前活性の幅が本当に頭打ちになるか。
2. 同時点の恒等式を当て直すだけでなく、前半の供給と削りから後半を予測できるか。
3. CondA の非定常入力の変化量、最適化器、Adam ε が実効変位と幅をどう変えるか。
4. 素のノルムと幅の乖離が別の環境にもあるか。

恒等式の残差が小さいこと、同じ窓で計算した W=D/(2|c|) の一致は独立な支持に数えない。
W に対する D の回帰傾きだけから因果の一方通行・独立性を結論しない。

## 2. 実験群（性能を見て選び直さない）

共通: 既存データを読み取り、作業は専用 worktree。新規 seed は100–104、5本すべてを残す。no weight decay / no parameter reset。

### A. CondA、新規

m=20、persistent bits f=15、残り5bitは各step独立 Bernoulli(1/2)。LTU teacher hidden100、beta=.7、固定教師。learner20→100→1、leaky .1、既存 Kaiming 初期化、MSE（係数2の勾配）、online batch1。
100 tasks、通常 T=10000 steps/task、全腕同じ seed の初期値・教師・random bitsを共有。
境界で15bitのランダム順列の先頭k個を反転。k変更が入力乱数の消費を変えないよう stream を分離。k=0でもflip streamを消費。

| arm | optimizer | lr | epsilon | k | T |
|---|---|---:|---:|---:|---:|
| A_k0 | Adam | .001 | 1e-8 | 0 | 10000 |
| A_k1 | Adam | .001 | 1e-8 | 1 | 10000 |
| A_k3 | Adam | .001 | 1e-8 | 3 | 10000 |
| A_k7 | Adam | .001 | 1e-8 | 7 | 10000 |
| E_k1 | Adam | .001 | 1e-3 | 1 | 10000 |
| S_k1 | SGD | .01 | — | 1 | 10000 |
| A_fast | Adam | .001 | 1e-8 | 1 | 1000 |

Adam beta=(.9,.999)。全腕 weights/moments を持ち越す。A_fast は同じ100 tasksだが更新総数が異なるため、非定常度だけの効果とは呼ばない。kを動かしても教師の複雑度は変えない。
各task末に W,b,v,c,flip_state と全32点上の MSE、前活性幅・平均を保存。initと100末尾の101 snapshots。境界変更前の末尾を保存する。

### B. RL-MNIST、新規

784→100→100→10、leaky .1、既存 pmnist_rlmnist_0906 の入力・初期化・label RNG・batch順の定義。
seed100–104、固定1200画像、taskごと独立uniform10class再ラベル、Adam lr=.001, beta=(.9,.999), eps=1e-8。
50 tasks、各400 epochs×75 batches×batch16=30000更新。既存のcanonical設定から短縮しない。
同設定 ELU を比較として追加するのは、数値結果を読まない速度smokeで全生産処理の予測所要時間が4時間以内と判明した場合のみ。採否を実行前のrun_manifestに記録。

### C. PermutedMNIST、新規

同じMLPとseed100–104、leaky .1、既存 pmnist_0905 のcanonical設定。
taskごとstratified10000画像、pixel permutation、SGD lr=.01、batch16、1 epoch=625更新、50 tasks。
各taskの実入力を復元できるsubset indices/permutationを保存。初期状態と全task末の全重み、accuracy/online accuracyを記録。

### D. CIFAR参照（既存・事後）

rlcifar_mlp_battle_0918 の LR raw/std、seed0–4、t00–t50の保存済みWを参照。新規性・独立再現の件数に数えない。
元のeffdisp_cifar_0923は既知の別走なので、数値の完全一致は要求しない。第一層のみ。

## 3. 計量と時間窓

行列Wは第一層、D_t=W_t−W_(t−1)。同じ入力共分散Σに対して V=tr(WΣWᵀ), Q=tr(DΣDᵀ), X=tr(WΣDᵀ)。
ΔV=Q+2X、c=X/sqrt(VQ)、rho=sqrt(Q/V)、window balance B=−2sum(X)/sum(Q)。
必ず前時点Wを使う。task末→次task末の同じlagのみ。V,Q,Xはfloat64で計算。
Q=0/V=0 は無効・停止として明示し、NaNを0に置換しない。極小Qの停止を平衡に数えない。

CondA: Σ=diag(0×15,.25×5) は各taskの中心化共分散で一定。persistent成分は幅に入らない。µ_t と b から mean preactivation=Wµ_t+b を別記し、意味のある平均変化を死重と呼ばない。
RL-MNIST/CIFAR: seedの固定画像bankから中心化共分散（population、分母N）を使う。
PM: 第一taskの入力bankを固定した reference metric と、各task実入力の actual width を併記。後者はΣ_tが動くため

V_t(Σ_t)−V_(t−1)(Σ_(t−1)) = Q_t(Σ_(t−1))+2X_t(Σ_(t−1)) + tr[W_t(Σ_t−Σ_(t−1))W_tᵀ]

を使う。最後のcovariance項を削りへ混ぜない。固定referenceでの平衡を、actual width の平衡と言い換えない。PMの二つのmetricはすべて掲載する。
素の計量Σ=Iも全群で記録。共分散の零空間と小さい正固有値を区別する。層2は今回は主判定対象外。

## 4. 登録判定

seed単位の値を先に算出。群の代表はseed中央値。5seedのうち4seed以上で通った場合のみ群PASS。CIはseedを単位としたbootstrap95%（5000 draws、固定seed924）。unitを独立seed扱いしない。
late = 最後20%のtransitions（最低10）。t0→t1はlateにもfitにも入れない。

P1 NEAR_BALANCE: late B∈[.9,1.1]、late sumQ/meanV≥.1、c<0のtransition割合≥.8。
P2 PLATEAU: lateで logV対logtask の傾きの絶対値≤.2、late後半meanV/前半meanV∈[.9,1.1]、P1も通る。
P1のみなら「更新量に対して収支差が小さい」、P2まで通っても観測地平内の頭打ちと書く。有限系列から漸近的定常性は証明しない。
STOPPED: late sumQ/meanV<.1。ELU等で全滅/精度低下があればそれも併記。停止によるplateauをactive equilibriumと呼ばない。

P3 HELDOUT_CLOSURE: calibrationはtask6..floor(T/2)、testは残り。q=mean(Q), gamma=−sum(X)/sum(V)をcalibrationだけで推定。
観測したsplit時点Vから Vhat_(t+1)=(1−2gamma)Vhat_t+q を再帰し、testの観測Q,X,Vを再入力しない。
0<gamma<1、全予測が正、test MAPE≤.2、固定split値のpersistence予測よりMAPEが20%以上低いときPASS。
両MAPE<.05は「ほぼ動かず予測モデルを識別できない」(UNINFORMATIVE)。負予測・不安定係数・不足点を隠さない。
このclosureは元の定数D,c説そのものではなく、独立に予測力を調べる簡単な線形収支モデル。元の比の一致とは別の判定。

P4 RAW_VS_EFFECTIVE: late raw logノルム二乗の傾き−effectiveの傾き≥.2、かつeffective P2 PASSのときSEPARATED。そうでなければNOT_ESTABLISHED。
P5 CondA非定常度: A_k7/A_k1 のlate実効変位 sqrtQ のpaired中央値比>1.2を予測。幅の単調増大は予測しない（削りも変わる）。k0/1/3/7の実測は全て掲載。
P6 optimizer/ε: A_k1のsqrtQがS_k1より大きく、E_k1がA_k1より小さいという方向を予測。5seed中4seed以上で両方向が揃えばPASS。学習率を合わせていないためoptimizer固有の普遍倍率とは呼ばない。
W対Dのlog回帰傾き、unit別c/収支はreport-only。傾き<1だけで固定点の存在・因果性を主張しない。

## 5. 実行前の予測（Codex）

- RL-MNIST leakyはP1を満たす可能性が高い（主観70%）。P2の厳しい頭打ちは50%。元CIFARも単なる減速が残っている可能性を残す。
- PMではreferenceとactualが乖離しうる。actual幅の変化を説明するにはcovariance項が必要で、元の一定Σの式だけの一般化は通らないと予測する。
- CondA k>0で負のXが出るが、cの絶対値がCIFARの.24–.35に揃うとは予測しない。A_k7はA_k1よりDが大きい（70%）、幅は供給と削りの競争なので方向未指定。
- k0は変位が小さくなる。定常入力でもオンライン勾配雑音は残り得るので、c=0や「平衡が存在しない」は予測しない。
- εを増すとDが減る（70%）。幅も下がる可能性はあるが削りの同時変化次第。SGDとの比較は学習進度の違いも残る。
- 「WがDにほぼ影響しない」という強い因果主張は、今回の観察だけでは確定しない。P3の不成立も十分あり得る（50%）。

## 6. 実装・資源・理論への分岐

単純実装はgpt-6-sol。数式・主判定・結果解釈はrootが確認。勾配/Adamを既存実装またはautogradで照合、固定/可変Σ恒等式、Qゼロ、集約順、heldout漏れ、k個反転、境界時刻を検査。
smokeは/tmp、2tasksまで・速度と数値健全性のみ。productionと混ぜない。計算は既存ローカルGPU/CPUのみ。所要時間予測4時間を超えるとELUを省く。主3環境は優先し、5seed固定・事後の都合で途中seedを削らない。非有限値の腕はFAILとして停止し勝手にlrを変えない。
実行状態の確認は10分間隔、状態不変の連投はしない。生データはobsidian-research-data/effdisp_validation_0924へ退避しSHA256 manifestを残す。
P1/P2/P3が複数環境で支持されれば、供給/削りの条件付きモデルと固定点の安定性を理論化する。不支持でも反例と適用条件を記述する。恒等式から因果を導いたことにはしない。新しい実験を無制限に追加しない。
